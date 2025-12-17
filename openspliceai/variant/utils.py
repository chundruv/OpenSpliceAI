from importlib.resources import files
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import os, glob
from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.constants import *
from openspliceai.predict.predict import *
from openspliceai.predict.utils import *
    
##############################################
## LOADING PYTORCH AND KERAS MODELS
##############################################

def setup_device():
        """Select computation device based on availability."""
        device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
        return torch.device(device_str)

def load_pytorch_models(model_path, CL):
    """
    Loads a SpliceAI PyTorch model from given state, inferring device.
    
    Params:
    - model_path (str): Path to the model state dict, or a directory of models
    - CL (int): Context length parameter for model conversion.
    
    Returns:
    - loaded_models (list): SpliceAI model(s) loaded with given state.
    """
    
    def load_model(device, flanking_size):
        """Loads the given model."""
        # Hyper-parameters:
        # L: Number of convolution kernels
        # W: Convolution window size in each residual unit
        # AR: Atrous rate in each residual unit
        L = 32
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        N_GPUS = 2
        BATCH_SIZE = 18*N_GPUS

        if int(flanking_size) == 80:
            W = np.asarray([11, 11, 11, 11])
            AR = np.asarray([1, 1, 1, 1])
            BATCH_SIZE = 18*N_GPUS
        elif int(flanking_size) == 400:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
            BATCH_SIZE = 18*N_GPUS
        elif int(flanking_size) == 2000:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                            21, 21, 21, 21])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10])
            BATCH_SIZE = 12*N_GPUS
        elif int(flanking_size) == 10000:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                            21, 21, 21, 21, 41, 41, 41, 41])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10, 25, 25, 25, 25])
            BATCH_SIZE = 6*N_GPUS

        CL = 2 * np.sum(AR*(W-1))

        print(f"\t[INFO] Context nucleotides {CL}")
        print(f"\t[INFO] Sequence length (output): {SL}")
        
        model = SpliceAI(L, W, AR).to(device)
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

        return model, params
    
    # Setup device
    device = setup_device()
    
    # Load all model state dicts given the supplied model path
    if os.path.isdir(model_path):
        model_files = glob.glob(os.path.join(model_path, '*.p[th]')) # gets all PyTorch models from supplied directory
        if not model_files:
            logging.error(f"No PyTorch model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = torch.load(model_file, map_location=device)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading PyTorch model from file {model_file}: {e}. Skipping...")
                
        if not models:
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
            exit()
    
    elif os.path.isfile(model_path):
        try:
            models = [torch.load(model_path, map_location=device)]
        except Exception as e:
            logging.error(f"Error loading PyTorch model from file {model_path}: {e}.")
            exit()
        
    else:
        logging.error(f"Invalid path: {model_path}")
        exit()
    
    # Load state of model to device
    # NOTE: supplied model paths should be state dicts, not model files  
    loaded_models = []
    
    for state_dict in models:
        try: 
            model, params = load_model(device, CL) # loads new SpliceAI model with correct hyperparams
            model.load_state_dict(state_dict)      # loads state dict
            model = model.to(device)               # puts model on device
            model.eval()                           # puts model in evaluation mode
            loaded_models.append(model)            # appends model to list of loaded models  
        except Exception as e:
            logging.error(f"Error processing model for device: {e}. Skipping...")
            
    if not loaded_models:
        logging.error("No models were successfully loaded to the device.")
        exit()
        
    return loaded_models

def load_keras_models(model_path):
    """
    Loads Keras models from given path.
    
    Params:
    - model_path (str): Path to the model file or directory of models.
    
    Returns:
    - models (list): List of loaded Keras models.
    """
    from tensorflow import keras
    
    if os.path.isdir(model_path): # directory supplied
        model_files = glob.glob(os.path.join(model_path, '*.h5')) # get all Keras models from a directory
        if not model_files:
            logging.error(f"No Keras model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = keras.models.load_model(model_file)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading Keras model from file {model_file}: {e}. Skipping...")
                
        if not models:
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
            exit()
            
        return models
    
    elif os.path.isfile(model_path): # file supplied
        try:
            return [keras.models.load_model(model_path)]
        except Exception as e:
            logging.error(f"Error loading Keras model from file {model_path}: {e}")
            exit()
        
    else: # invalid path
        logging.error(f"Invalid path: {model_path}")
        exit()

##############################################
## FORMATTING INPUT DATA FOR PREDICTION
##############################################

def one_hot_encode(seq):
    """
    One-hot encode a DNA sequence.
    
    Args:
        seq (str): DNA sequence to be encoded.
    
    Returns:
        np.ndarray: One-hot encoded representation of the sequence.
    """

    # Define a mapping matrix for nucleotide to one-hot encoding
    map = np.asarray([[0, 0, 0, 0],  # N or any invalid character
                      [1, 0, 0, 0],  # A
                      [0, 1, 0, 0],  # C
                      [0, 0, 1, 0],  # G
                      [0, 0, 0, 1]]) # T

    # Replace nucleotides with corresponding indices
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    # Convert the sequence to one-hot encoded numpy array
    return map[np.fromstring(seq, np.int8) % 5]


####################################################################################################################################
#######                                                                                                                      #######
#######                                             ANNOTATOR CLASS                                                          #######
#######                                                                                                                      #######
####################################################################################################################################

class Annotator:
    """
    Annotator class to handle gene annotations and reference sequences.
    It initializes with the reference genome, annotation data, and optional model configuration.
    """
    
    def __init__(self, ref_fasta, annotations, model_path='SpliceAI', model_type='keras', CL=80):
        """
        Initializes the Annotator with reference genome, annotations, and model settings.
        
        Args:
            ref_fasta (str): Path to the reference genome FASTA file.
            annotations (str): Path or name of the annotation file (e.g., 'grch37', 'grch38').
            model_path (str, optional): Path to the model file or type of model ('SpliceAI'). Defaults to SpliceAI.
            model_type (str, optional): Type of model ('keras' or 'pytorch'). Defaults to 'keras'.
            CL (int, optional): Context length parameter for model conversion. Defaults to 80.
        """

        # Load annotation file based on provided annotations type
        if annotations == 'grch37':
            annotations = './data/vcf/grch37.txt'
        elif annotations == 'grch38':
            annotations = './data/vcf/grch38.txt'

        # Load and parse the annotation file
        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            # Extract relevant columns into numpy arrays for efficient access
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy() + 1  # Transcription start sites (1-based indexing)
            self.tx_ends = df['TX_END'].to_numpy()  # Transcription end sites
            
            # Extract and process exon start and end sites, convert into numpy array format
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i]) + 1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e)) 
            exit()  # Exit if the file cannot be read
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()  # Exit if the file format is incorrect

        # Load the reference genome fasta file
        try:
            self.ref_fasta = Fasta(ref_fasta, sequence_always_upper=True, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))  # Log file read error
            exit()  # Exit if the file cannot be read

        # Load models based on the specified model type or file
        if model_path == 'SpliceAI':
            from tensorflow import keras
            paths = ('./models/spliceai/spliceai{}.h5'.format(x) for x in range(1, 6))  # Generate paths for SpliceAI models
            self.models = [keras.models.load_model(x) for x in paths]
            self.keras = True
        elif model_type == 'keras': # load models using keras
            self.models = load_keras_models(model_path)
            self.keras = True
        elif model_type == 'pytorch': # load models using pytorch 
            self.models = load_pytorch_models(model_path, CL)
            self.keras = False
        else:
            logging.error('Model type {} not supported'.format(model_type))
            exit()
        
        print(f'\t[INFO] {len(self.models)} model(s) loaded successfully')

    def get_name_and_strand(self, chrom, pos):
        """
        Retrieve gene names and strands overlapping a given chromosome position.
        
        Args:
            chrom (str): Chromosome identifier.
            pos (int): Position on the chromosome.
        
        Returns:
            tuple: Lists of gene names, strands, and their indices overlapping the given position.
        """

        # Normalize chromosome identifier to match the annotation format
        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        # Find indices of annotations overlapping the given chromosome and position
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                             np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs  # Return matching gene names and strands
        else:
            return [], [], []  # Return empty lists if no matches are found

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        #dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos, key=abs)
        dist_exon_bdry = np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

        return dist_ann
    
##############################################
## CALCULATING DELTA SCORES
##############################################

def has_stop_codon(sequence):
    """Check if a DNA sequence contains a stop codon when translated.

    Args:
        sequence: DNA sequence string

    Returns:
        bool: True if sequence contains a stop codon
    """
    if not sequence or len(sequence) < 3:
        return False

    # Standard genetic code stop codons
    stop_codons = {'TAA', 'TAG', 'TGA'}

    # Check each codon in the sequence (reading frame starts at position 0)
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if codon in stop_codons:
            return True
    return False


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base.upper(), 'N') for base in reversed(sequence))


def find_stop_in_frame(sequence, frame_offset):
    """
    Find the first stop codon in a specific reading frame.

    Args:
        sequence: DNA sequence string
        frame_offset: Reading frame offset (1 or 2 for frameshift)

    Returns:
        Position of first stop codon in bp from start, or None if no stop found
    """
    if not sequence or len(sequence) < 3:
        return None

    stop_codons = {'TAA', 'TAG', 'TGA'}

    # Start reading from frame_offset
    for i in range(frame_offset, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if codon in stop_codons:
            return i  # Return position in bp
    return None

def normalise_chrom(source, target):
    """
    Normalize chromosome identifiers to ensure consistency in format (with or without 'chr' prefix).
    
    Args:
        source (str): Source chromosome identifier.
        target (str): Target chromosome identifier for comparison.
    
    Returns:
        str: Normalized chromosome identifier.
    """

    def has_prefix(x):
        return x.startswith('chr')  # Check if a chromosome name has 'chr' prefix

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')  # Remove 'chr' prefix if target doesn't have it
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source  # Add 'chr' prefix if target has it

    return source  # Return source as is if both or neither have 'chr' prefix

def get_delta_scores(record, ann, dist_var, mask, flanking_size=10000, precision=2):
    """
    Calculate delta scores for variant impacts on splice sites.
    
    Args:
        record (pysam Record): Record containing variant information (e.g., chrom, pos, ref, alts).
        ann (Annotator): Annotator instance with annotation and reference genome data.
        dist_var (int): Max distance between variant and gained/lost splice site, defaults to 50.
        mask (bool): Mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss, defaults to 0.
        flanking_size (int, optional): Size of the flanking region around the variant, defaults to 10000.
    
    Returns:
        list: Delta scores indicating the impact of variants on splicing.
    """

    # Define coverage and window size around the variant
    cov = 2 * dist_var + 1
    wid = flanking_size + cov
    delta_scores = []
    device = setup_device()

    # Validate the record fields
    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return delta_scores

    # Get gene names and strands overlapping the variant position
    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores  # Return empty list if no overlapping genes are found

    # Normalize chromosome and retrieve reference sequence around the variant
    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_scores

    # Check if the reference sequence matches the expected reference allele
    if seq[wid // 2 : wid // 2 + len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return delta_scores

    # Check if the sequence length matches the expected window size
    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return delta_scores

    # Skip records with a reference allele longer than the distance variable
    if len(record.ref) > 2 * dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return delta_scores

    # Iterate over each alternative allele and each gene index to calculate delta score
    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            # Skip specific alternative allele types
            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue
            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            # Handle multi-nucleotide variants
            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                continue

            # Calculate position-related distances
            #dist_ann = ann.get_pos_data(idxs[i], record.pos)
            dist_ann_all = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid // 2 + dist_ann_all[0], 0), max(wid // 2 - dist_ann_all[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)
            cov_half = cov // 2

            # Construct reference and alternative sequences with padding
            x_ref = 'N' * pad_size[0] + seq[pad_size[0]: wid - pad_size[1]] + 'N' * pad_size[1]
            x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]

            # One-hot encode the sequences
            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            '''separate handling of PyTorch and Keras models'''
            if ann.keras: # keras model handling
                # Reverse the sequences if on the negative strand
                if strands[i] == '-':
                    x_ref = x_ref[:, ::-1, ::-1]
                    x_alt = x_alt[:, ::-1, ::-1]

                # Predict scores using the models
                y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(len(ann.models))], axis=0)
                y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(len(ann.models))], axis=0)
                
                # Reverse the predicted scores if on the negative strand
                if strands[i] == '-':
                    y_ref = y_ref[:, ::-1]
                    y_alt = y_alt[:, ::-1]
                    
            else: # pytorch model handling
                
                # Reshape tensor to match the model input shape
                x_ref = x_ref.transpose(0, 2, 1)
                x_alt = x_alt.transpose(0, 2, 1)
                
                # Convert to PyTorch tensors
                x_ref = torch.tensor(x_ref, dtype=torch.float32)
                x_alt = torch.tensor(x_alt, dtype=torch.float32)

                # Reverse the sequences if on the negative strand
                if strands[i] == '-':
                    x_ref = torch.flip(x_ref, dims=[1, 2])
                    x_alt = torch.flip(x_alt, dims=[1, 2])

                # Put tensors on device
                x_ref = x_ref.to(device)
                x_alt = x_alt.to(device)
                
                # Predict scores using the models
                with torch.no_grad():
                    y_ref = torch.mean(torch.stack([ann.models[m](x_ref).detach().cpu() for m in range(len(ann.models))]), axis=0)
                    y_alt = torch.mean(torch.stack([ann.models[m](x_alt).detach().cpu() for m in range(len(ann.models))]), axis=0)
                
                # Remove flanking sequence and permute shape
                y_ref = y_ref.permute(0, 2, 1)
                y_alt = y_alt.permute(0, 2, 1)

                # Reverse the predicted scores if on the negative strand and convert back to numpy arrays
                if strands[i] == '-':
                    y_ref = torch.flip(y_ref, dims=[1])
                    y_alt = torch.flip(y_alt, dims=[1])
                
                # Convert to numpy arrays
                y_ref = y_ref.numpy()
                y_alt = y_alt.numpy()
            '''end'''
            
            # Manually crop the output if it exceeds the expected coverage window
            # This handles cases where the model output is not automatically cropped to the target window
            if y_ref.shape[1] > cov:
                start_idx = wid // 2 - cov // 2
                y_ref = y_ref[:, start_idx : start_idx + cov, :]
                y_alt = y_alt[:, start_idx : start_idx + cov + alt_len - ref_len, :]

            # Adjust the alternative sequence scores based on reference and alternative lengths
            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2 + alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov // 2 + alt_len:]
                ], axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2],
                    np.max(y_alt[:, cov // 2 : cov // 2 + alt_len], axis=1)[:, None, :],
                    y_alt[:, cov // 2 + alt_len:]
                ], axis=1)

            seq_offset = record.pos - wid // 2 - 1
            def get_subsequence(start_genomic, end_genomic):
                """Extract subsequence from pre-fetched seq given genomic coordinates."""
                start_idx = start_genomic - seq_offset
                end_idx = end_genomic - seq_offset
                if start_idx < 0 or end_idx > len(seq):
                    return None
                return seq[start_idx:end_idx]
            # Concatenate the reference and alternative scores
            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

            dist_ann_set = set(dist_ann_all[2])
            mask_pa = mask and ((idx_pa - cov_half) in dist_ann_set)
            mask_na = mask and ((idx_na - cov_half) not in dist_ann_set)
            mask_pd = mask and ((idx_pd - cov_half) in dist_ann_set)
            mask_nd = mask and ((idx_nd - cov_half) not in dist_ann_set)

            # Extract raw score for all annotated splice sites (MANE splice sites within context)
            ann_acpt_ss = [1000, 0]  # [position, ref_score]
            ann_donor_ss = [1000, 0]
            mane_parts = {'Acceptor': [], 'Donor': []}  # Build string directly for speed
            # Note: cov_half already calculated at line 354, no need to recalculate

            # Pre-filter sites within context window for efficiency
            for i in dist_ann_all[2]:
                abs_i = abs(i)
                if abs_i < cov_half:
                    idx = cov_half + i
                    # Cache array lookups to avoid repeated indexing
                    donor_ref = y[0, idx, 2]
                    acceptor_ref = y[0, idx, 1]
                    donor_alt = y[1, idx, 2]
                    acceptor_alt = y[1, idx, 1]

                    if donor_ref > acceptor_ref:
                        # Donor site - use cached values
                        mane_parts['Donor'].append(f"{i},{donor_ref:.2f},{donor_alt:.2f}")
                        if abs_i < abs(ann_donor_ss[0]):
                            ann_donor_ss = [i, donor_ref]
                    else:
                        # Acceptor site - use cached values
                        mane_parts['Acceptor'].append(f"{i},{acceptor_ref:.2f},{acceptor_alt:.2f}")
                        if abs_i < abs(ann_acpt_ss[0]):
                            ann_acpt_ss = [i, acceptor_ref]

            # Extract all sites with raw score above 0.5 - optimized with vectorized operations
            sites_gt05_parts = {'Acceptor': [], 'Donor': []}

            # Process all at once for better performance
            acceptor_ref_mask = y[0, :, 1] > 0.5
            donor_ref_mask = y[0, :, 2] > 0.5
            acceptor_alt_mask = y[1, :, 1] > 0.5
            donor_alt_mask = y[1, :, 2] > 0.5

            # Build strings directly - cache array values to avoid repeated indexing
            for pos in np.flatnonzero(np.logical_or(acceptor_ref_mask, acceptor_alt_mask)):
                ref_score = y[0, pos, 1]
                alt_score = y[1, pos, 1]
                sites_gt05_parts['Acceptor'].append(f"{pos - cov_half},{ref_score:.2f},{alt_score:.2f}")
            for pos in np.flatnonzero(np.logical_or(donor_ref_mask, donor_alt_mask)):
                ref_score = y[0, pos, 2]
                alt_score = y[1, pos, 2]
                sites_gt05_parts['Donor'].append(f"{pos - cov_half},{ref_score:.2f},{alt_score:.2f}")

            # Pre-compute length once for both donor and acceptor processing
            dist_ann_len = len(dist_ann_all[2])

            # Process DONOR splice changes - optimized to cache index lookups
            donor_frame_change = '.'
            splice_change_donor = 'NoDonorChange'
            aa_change_donor = '.'
            donor_max_btwn_ss_pos = None
            donor_max_btwn_ss_score = 0
            donor_index = -1
            left_bound = right_bound = None

            if ann_donor_ss[0] < 1000:
                # Find the donor index once and cache it
                donor_index = np.searchsorted(dist_ann_all[2], ann_donor_ss[0])

                # Get bounds for search window (between previous and next annotated splice sites)
                if donor_index > 0 and donor_index < dist_ann_len - 1:
                    left_bound = cov_half + dist_ann_all[2][donor_index-1] + 1
                    right_bound = cov_half + dist_ann_all[2][donor_index+1]
                    # Bounds checking for array access
                    left_bound = max(0, min(left_bound, y.shape[1] - 1))
                    right_bound = max(left_bound + 1, min(right_bound, y.shape[1]))
                    if right_bound > left_bound:
                        donor_max_btwn_ss_pos = left_bound + y[1, left_bound:right_bound, 2].argmax() - cov_half
                    else:
                        donor_max_btwn_ss_pos = y[1, :, 2].argmax() - cov_half
                else:
                    donor_max_btwn_ss_pos = y[1, :, 2].argmax() - cov_half
            else:
                donor_max_btwn_ss_pos = y[1, :, 2].argmax() - cov_half

            # Safe array access with bounds checking
            max_pos_idx = cov_half + donor_max_btwn_ss_pos
            max_pos_idx = max(0, min(max_pos_idx, y.shape[1] - 1))
            donor_max_btwn_ss_score = y[1, max_pos_idx, 2]

            # Check for competing sites in REF (within 5% of score) - with safety check
            if ann_donor_ss[0] < 1000 and ann_donor_ss[1] > 0.01 and left_bound is not None:
                # Safe division: only check if reference score is substantial (> 0.01)
                ref_scores = y[0, left_bound:right_bound, 2]
                ref_competing = np.sum(np.abs(ref_scores - ann_donor_ss[1]) / ann_donor_ss[1] < 0.05) > 1
                if ref_competing:
                    splice_change_donor = 'NoDonorChange,CompetingSitesInRef'

            # Detect donor changes
            if donor_max_btwn_ss_pos != ann_donor_ss[0] and donor_max_btwn_ss_score > 0.5:
                if ann_donor_ss[0] < 1000:
                    if left_bound is not None and donor_max_btwn_ss_score > 0.01:
                        # Safe division: only check if score is substantial
                        alt_scores = y[1, left_bound:right_bound, 2]
                        alt_competing = np.sum(np.abs(alt_scores - donor_max_btwn_ss_score) / donor_max_btwn_ss_score < 0.05) > 1

                        splice_change_donor = 'DonorChange'
                        if alt_competing:
                            splice_change_donor += ',CompetingSitesInALT'

                        # Calculate frame change and AA difference
                        position_diff = abs(donor_max_btwn_ss_pos - ann_donor_ss[0])
                        donor_frame_change = position_diff % 3

                        # Calculate AA change for splice changes with stop codon checking
                        if donor_frame_change == 0:
                            aa_diff = position_diff // 3
                            sign = '+' if donor_max_btwn_ss_pos > ann_donor_ss[0] else '-'

                            # Check for stop codons in the shifted region
                            has_stop = False
                            if aa_diff > 0:
                                try:
                                    # Calculate positions for the sequence change
                                    start_pos = min(record.pos + ann_donor_ss[0], record.pos + donor_max_btwn_ss_pos)
                                    end_pos = max(record.pos + ann_donor_ss[0], record.pos + donor_max_btwn_ss_pos)
                                    sequence = get_subsequence(start_pos, end_pos)

                                    if sequence and strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    if sequence:
                                        has_stop = has_stop_codon(sequence)
                                except:
                                    pass

                            # Format output based on whether it's an addition or deletion
                            if donor_max_btwn_ss_pos > ann_donor_ss[0]:  # Addition
                                if has_stop:
                                    aa_change_donor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,+STOP)"
                                else:
                                    aa_change_donor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,NoStop)"
                            else:  # Deletion
                                if has_stop:
                                    aa_change_donor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,-STOP)"
                                else:
                                    aa_change_donor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,NoStopRemoved)"
                        else:
                            # Frameshift: check for stop codon within context window
                            stop_annotation = ''
                            try:
                                # Extract sequence from the splice site change to end of context window
                                start_pos = record.pos + max(ann_donor_ss[0], donor_max_btwn_ss_pos)
                                end_pos = record.pos + cov_half
                                sequence = get_subsequence(start_pos, end_pos)

                                if sequence:
                                    if strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    # Check for stop in the frameshift reading frame
                                    stop_pos = find_stop_in_frame(sequence, donor_frame_change)
                                    if stop_pos is not None:
                                        stop_annotation = f",StopAt+{stop_pos}bp"
                            except:
                                pass

                            aa_change_donor = f"SpliceShift({position_diff}bp,frameshift={donor_frame_change}{stop_annotation})"
                else:
                    # Donor gain - find distance to nearest MANE donor site (search all MANE sites, not just in context)
                    splice_change_donor = 'NoDonorInRef,DonorGain'
                    closest_donor_dist = float('inf')
                    closest_donor_pos = None

                    # Search all MANE donor sites in dist_ann_all
                    if len(dist_ann_all[2]) > 0:
                        for ss_pos in dist_ann_all[2]:
                            # For sites outside context window, we can't check scores, so check both in and out of window
                            is_in_window = abs(ss_pos) < cov_half

                            if is_in_window:
                                idx = cov_half + ss_pos
                                if 0 <= idx < y.shape[1]:
                                    donor_score_ref = y[0, idx, 2]
                                    acceptor_score_ref = y[0, idx, 1]

                                    # Only consider donor sites
                                    if donor_score_ref > acceptor_score_ref:
                                        dist = abs(ss_pos - donor_max_btwn_ss_pos)
                                        if dist < closest_donor_dist and ss_pos != donor_max_btwn_ss_pos:
                                            closest_donor_dist = dist
                                            closest_donor_pos = ss_pos
                            else:
                                # Outside window - assume it's the type annotated in MANE (we can't check scores)
                                # Just find the closest one
                                dist = abs(ss_pos - donor_max_btwn_ss_pos)
                                if dist < closest_donor_dist and ss_pos != donor_max_btwn_ss_pos:
                                    closest_donor_dist = dist
                                    closest_donor_pos = ss_pos

                    if closest_donor_pos is not None:
                        distance_to_next = closest_donor_pos - donor_max_btwn_ss_pos
                        if distance_to_next > 0:
                            splice_change_donor = f'NoDonorInRef,DonorGain(NextDonor+{distance_to_next}bp)'
                        else:
                            splice_change_donor = f'NoDonorInRef,DonorGain(NextDonor{distance_to_next}bp)'
            elif ann_donor_ss[1] < 0.5:
                splice_change_donor = 'NoStrongDonorInRef'

            # Check for donor loss - reuse cached donor_index
            if ann_donor_ss[0] < 1000 and ann_donor_ss[1] > 0.5 and donor_index >= 0:
                # Cache array index to avoid repeated calculation
                donor_ss_idx = cov_half + ann_donor_ss[0]
                donor_loss_score = y[0, donor_ss_idx, 2] - y[1, donor_ss_idx, 2]
                if donor_loss_score > 0.2:  # Significant loss
                    has_compensating_acceptor_gain = donor_max_btwn_ss_score > 0.5 and 'DonorChange' in splice_change_donor

                    if not has_compensating_acceptor_gain and donor_index < dist_ann_len - 1:
                        # Update splice change status
                        splice_change_donor = 'DonorLoss'

                        # Calculate exon skipping vs intron retention
                        next_ss_pos = dist_ann_all[2][donor_index + 1]
                        distance_to_next = abs(next_ss_pos - ann_donor_ss[0])

                        # Intron retention - always calculate AA change and check for stop codons
                        intron_frame = distance_to_next % 3
                        intron_aa = distance_to_next // 3

                        # Extract sequence for intron retention to check for stop codons
                        has_stop = False
                        if intron_frame == 0 and distance_to_next >= 3:
                            try:
                                # Calculate absolute genomic positions
                                donor_abs_pos = record.pos + ann_donor_ss[0]
                                next_ss_abs_pos = record.pos + next_ss_pos
                                start_pos = min(donor_abs_pos, next_ss_abs_pos)
                                end_pos = max(donor_abs_pos, next_ss_abs_pos)

                                # Extract sequence from pre-fetched seq
                                sequence = get_subsequence(start_pos, end_pos)

                                # Reverse complement if on negative strand
                                if sequence and strands[i] == '-':
                                    sequence = reverse_complement(sequence)

                                # Check for stop codons
                                if sequence:
                                    has_stop = has_stop_codon(sequence)
                            except:
                                # If sequence extraction fails, continue without stop codon check
                                pass

                        if intron_frame == 0:
                            if has_stop:
                                intron_ret_str = f"IntronRetention(inframe,+{intron_aa}aa,+{distance_to_next}bp,+STOP)"
                            else:
                                intron_ret_str = f"IntronRetention(inframe,+{intron_aa}aa,+{distance_to_next}bp,NoStop)"
                        else:
                            # Frameshift: check for stop codon within context window
                            stop_annotation = ''
                            try:
                                # Extract sequence from donor to end of context window
                                start_pos = record.pos + ann_donor_ss[0]
                                end_pos = record.pos + cov_half
                                sequence = get_subsequence(start_pos, end_pos)

                                if sequence:
                                    if strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    stop_pos = find_stop_in_frame(sequence, intron_frame)
                                    if stop_pos is not None:
                                        stop_annotation = f",StopAt+{stop_pos}bp"
                            except:
                                pass

                            intron_ret_str = f"IntronRetention(+{intron_aa}aa,+{distance_to_next}bp,frameshift={intron_frame}{stop_annotation})"

                        # Exon skipping
                        if donor_index < dist_ann_len - 2:
                            exon_skip_distance = abs(dist_ann_all[2][donor_index + 2] - ann_donor_ss[0])
                            exon_skip_frame = exon_skip_distance % 3
                            exon_skip_aa = exon_skip_distance // 3

                            # Check for stop codons in exon skipping case (deletion)
                            has_stop_exon = False
                            if exon_skip_frame == 0 and exon_skip_distance >= 3:
                                try:
                                    skip_abs_pos = record.pos + dist_ann_all[2][donor_index + 2]
                                    donor_abs_pos = record.pos + ann_donor_ss[0]
                                    start_pos = min(donor_abs_pos, skip_abs_pos)
                                    end_pos = max(donor_abs_pos, skip_abs_pos)

                                    sequence = get_subsequence(start_pos, end_pos)
                                    if sequence and strands[i] == '-':
                                        sequence = reverse_complement(sequence)
                                    if sequence:
                                        has_stop_exon = has_stop_codon(sequence)
                                except:
                                    pass

                            if exon_skip_frame == 0:
                                if has_stop_exon:
                                    exon_skip_str = f"ExonSkip(inframe,-{exon_skip_aa}aa,-{exon_skip_distance}bp,-STOP)"
                                else:
                                    exon_skip_str = f"ExonSkip(inframe,-{exon_skip_aa}aa,-{exon_skip_distance}bp,NoStopRemoved)"
                            else:
                                # Frameshift: check for stop codon within context window
                                stop_annotation = ''
                                try:
                                    # Extract sequence from donor to end of context window
                                    start_pos = record.pos + ann_donor_ss[0]
                                    end_pos = record.pos + cov_half
                                    sequence = get_subsequence(start_pos, end_pos)

                                    if sequence:
                                        if strands[i] == '-':
                                            sequence = reverse_complement(sequence)

                                        stop_pos = find_stop_in_frame(sequence, exon_skip_frame)
                                        if stop_pos is not None:
                                            stop_annotation = f",StopAt+{stop_pos}bp"
                                except:
                                    pass

                                exon_skip_str = f"ExonSkip(-{exon_skip_aa}aa,-{exon_skip_distance}bp,frameshift={exon_skip_frame}{stop_annotation})"

                            aa_change_donor = f"{intron_ret_str},{exon_skip_str}"
                            # Use intron retention frame for FRAME_CHANGE field
                            donor_frame_change = intron_frame
                        else:
                            aa_change_donor = intron_ret_str
                            donor_frame_change = intron_frame

            # Process ACCEPTOR splice changes - optimized to cache index lookups
            acceptor_frame_change = '.'
            splice_change_acceptor = 'NoAcceptorChange'
            aa_change_acceptor = '.'
            acceptor_max_btwn_ss_pos = None
            acceptor_max_btwn_ss_score = 0
            acceptor_index = -1
            acc_left_bound = acc_right_bound = None

            if ann_acpt_ss[0] < 1000:
                # Find the acceptor index once and cache it
                acceptor_index = np.searchsorted(dist_ann_all[2], ann_acpt_ss[0])

                # Get bounds for search window
                if acceptor_index > 0 and acceptor_index < dist_ann_len - 1:
                    acc_left_bound = cov_half + dist_ann_all[2][acceptor_index-1] + 1
                    acc_right_bound = cov_half + dist_ann_all[2][acceptor_index+1]
                    # Bounds checking for array access
                    acc_left_bound = max(0, min(acc_left_bound, y.shape[1] - 1))
                    acc_right_bound = max(acc_left_bound + 1, min(acc_right_bound, y.shape[1]))
                    if acc_right_bound > acc_left_bound:
                        acceptor_max_btwn_ss_pos = acc_left_bound + y[1, acc_left_bound:acc_right_bound, 1].argmax() - cov_half
                    else:
                        acceptor_max_btwn_ss_pos = y[1, :, 1].argmax() - cov_half
                else:
                    acceptor_max_btwn_ss_pos = y[1, :, 1].argmax() - cov_half
            else:
                acceptor_max_btwn_ss_pos = y[1, :, 1].argmax() - cov_half

            # Safe array access with bounds checking
            max_pos_idx = cov_half + acceptor_max_btwn_ss_pos
            max_pos_idx = max(0, min(max_pos_idx, y.shape[1] - 1))
            acceptor_max_btwn_ss_score = y[1, max_pos_idx, 1]

            # Check for competing sites in REF - with safety check
            if ann_acpt_ss[0] < 1000 and ann_acpt_ss[1] > 0.01 and acc_left_bound is not None:
                # Safe division: only check if reference score is substantial (> 0.01)
                ref_scores = y[0, acc_left_bound:acc_right_bound, 1]
                ref_competing = np.sum(np.abs(ref_scores - ann_acpt_ss[1]) / ann_acpt_ss[1] < 0.05) > 1
                if ref_competing:
                    splice_change_acceptor = 'NoAcceptorChange,CompetingSitesInRef'

            # Detect acceptor changes
            if acceptor_max_btwn_ss_pos != ann_acpt_ss[0] and acceptor_max_btwn_ss_score > 0.5:
                if ann_acpt_ss[0] < 1000:
                    if acc_left_bound is not None and acceptor_max_btwn_ss_score > 0.01:
                        # Safe division: only check if score is substantial
                        alt_scores = y[1, acc_left_bound:acc_right_bound, 1]
                        alt_competing = np.sum(np.abs(alt_scores - acceptor_max_btwn_ss_score) / acceptor_max_btwn_ss_score < 0.05) > 1

                        splice_change_acceptor = 'AcceptorChange'
                        if alt_competing:
                            splice_change_acceptor += ',CompetingSitesInALT'

                        # Calculate frame change and AA difference
                        position_diff = abs(acceptor_max_btwn_ss_pos - ann_acpt_ss[0])
                        acceptor_frame_change = position_diff % 3

                        # Calculate AA change for splice changes with stop codon checking
                        if acceptor_frame_change == 0:
                            aa_diff = position_diff // 3
                            sign = '+' if acceptor_max_btwn_ss_pos > ann_acpt_ss[0] else '-'

                            # Check for stop codons in the shifted region
                            has_stop = False
                            if aa_diff > 0:
                                try:
                                    # Calculate positions for the sequence change
                                    start_pos = min(record.pos + ann_acpt_ss[0], record.pos + acceptor_max_btwn_ss_pos)
                                    end_pos = max(record.pos + ann_acpt_ss[0], record.pos + acceptor_max_btwn_ss_pos)
                                    sequence = get_subsequence(start_pos, end_pos)

                                    if sequence and strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    if sequence:
                                        has_stop = has_stop_codon(sequence)
                                except:
                                    pass

                            # Format output based on whether it's an addition or deletion
                            if acceptor_max_btwn_ss_pos > ann_acpt_ss[0]:  # Addition
                                if has_stop:
                                    aa_change_acceptor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,+STOP)"
                                else:
                                    aa_change_acceptor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,NoStop)"
                            else:  # Deletion
                                if has_stop:
                                    aa_change_acceptor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,-STOP)"
                                else:
                                    aa_change_acceptor = f"SpliceShift(inframe,{sign}{aa_diff}aa,{sign}{position_diff}bp,NoStopRemoved)"
                        else:
                            # Frameshift: check for stop codon within context window
                            stop_annotation = ''
                            try:
                                # Extract sequence from the splice site change to end of context window
                                start_pos = record.pos + max(ann_acpt_ss[0], acceptor_max_btwn_ss_pos)
                                end_pos = record.pos + cov_half
                                sequence = get_subsequence(start_pos, end_pos)

                                if sequence:
                                    if strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    # Check for stop in the frameshift reading frame
                                    stop_pos = find_stop_in_frame(sequence, acceptor_frame_change)
                                    if stop_pos is not None:
                                        stop_annotation = f",StopAt+{stop_pos}bp"
                            except:
                                pass

                            aa_change_acceptor = f"SpliceShift({position_diff}bp,frameshift={acceptor_frame_change}{stop_annotation})"
                else:
                    # Acceptor gain - find distance to nearest MANE acceptor site (search all MANE sites, not just in context)
                    splice_change_acceptor = 'NoAcceptorInRef,AcceptorGain'
                    closest_acceptor_dist = float('inf')
                    closest_acceptor_pos = None

                    # Search all MANE acceptor sites in dist_ann_all
                    if len(dist_ann_all[2]) > 0:
                        for ss_pos in dist_ann_all[2]:
                            # For sites outside context window, we can't check scores
                            is_in_window = abs(ss_pos) < cov_half

                            if is_in_window:
                                idx = cov_half + ss_pos
                                if 0 <= idx < y.shape[1]:
                                    donor_score_ref = y[0, idx, 2]
                                    acceptor_score_ref = y[0, idx, 1]

                                    # Only consider acceptor sites
                                    if acceptor_score_ref > donor_score_ref:
                                        dist = abs(ss_pos - acceptor_max_btwn_ss_pos)
                                        if dist < closest_acceptor_dist and ss_pos != acceptor_max_btwn_ss_pos:
                                            closest_acceptor_dist = dist
                                            closest_acceptor_pos = ss_pos
                            else:
                                # Outside window - assume it's the type annotated in MANE
                                dist = abs(ss_pos - acceptor_max_btwn_ss_pos)
                                if dist < closest_acceptor_dist and ss_pos != acceptor_max_btwn_ss_pos:
                                    closest_acceptor_dist = dist
                                    closest_acceptor_pos = ss_pos

                    if closest_acceptor_pos is not None:
                        distance_to_next = closest_acceptor_pos - acceptor_max_btwn_ss_pos
                        if distance_to_next > 0:
                            splice_change_acceptor = f'NoAcceptorInRef,AcceptorGain(NextAcceptor+{distance_to_next}bp)'
                        else:
                            splice_change_acceptor = f'NoAcceptorInRef,AcceptorGain(NextAcceptor{distance_to_next}bp)'
            elif ann_acpt_ss[1] < 0.5:
                splice_change_acceptor = 'NoStrongAcceptorInRef'

            # Check for acceptor loss - reuse cached acceptor_index
            if ann_acpt_ss[0] < 1000 and ann_acpt_ss[1] > 0.5 and acceptor_index > 0:
                # Cache array index to avoid repeated calculation
                acceptor_ss_idx = cov_half + ann_acpt_ss[0]
                acceptor_loss_score = y[0, acceptor_ss_idx, 1] - y[1, acceptor_ss_idx, 1]
                if acceptor_loss_score > 0.2:  # Significant loss
                    has_compensating_donor_gain = acceptor_max_btwn_ss_score > 0.5 and 'AcceptorChange' in splice_change_acceptor

                    if not has_compensating_donor_gain and acceptor_index < dist_ann_len:
                        # Update splice change status
                        splice_change_acceptor = 'AcceptorLoss'

                        # Calculate exon skipping vs intron retention
                        prev_ss_pos = dist_ann_all[2][acceptor_index - 1]
                        distance_to_prev = abs(ann_acpt_ss[0] - prev_ss_pos)

                        # Intron retention - always calculate AA change and check for stop codons
                        intron_frame = distance_to_prev % 3
                        intron_aa = distance_to_prev // 3

                        # Extract sequence for intron retention to check for stop codons
                        has_stop = False
                        if intron_frame == 0 and distance_to_prev >= 3:
                            try:
                                # Calculate absolute genomic positions
                                acceptor_abs_pos = record.pos + ann_acpt_ss[0]
                                prev_ss_abs_pos = record.pos + prev_ss_pos
                                start_pos = min(acceptor_abs_pos, prev_ss_abs_pos)
                                end_pos = max(acceptor_abs_pos, prev_ss_abs_pos)

                                # Extract sequence from pre-fetched seq
                                sequence = get_subsequence(start_pos, end_pos)

                                # Reverse complement if on negative strand
                                if sequence and strands[i] == '-':
                                    sequence = reverse_complement(sequence)

                                # Check for stop codons
                                if sequence:
                                    has_stop = has_stop_codon(sequence)
                            except:
                                # If sequence extraction fails, continue without stop codon check
                                pass

                        if intron_frame == 0:
                            if has_stop:
                                intron_ret_str = f"IntronRetention(inframe,+{intron_aa}aa,+{distance_to_prev}bp,STOPAdded)"
                            else:
                                intron_ret_str = f"IntronRetention(inframe,+{intron_aa}aa,+{distance_to_prev}bp,NoStopAdded)"
                        else:
                            # Frameshift: check for stop codon within context window
                            stop_annotation = ''
                            try:
                                # Extract sequence from acceptor to end of context window
                                start_pos = record.pos + ann_acpt_ss[0]
                                end_pos = record.pos + cov_half
                                sequence = get_subsequence(start_pos, end_pos)

                                if sequence:
                                    if strands[i] == '-':
                                        sequence = reverse_complement(sequence)

                                    stop_pos = find_stop_in_frame(sequence, intron_frame)
                                    if stop_pos is not None:
                                        stop_annotation = f",StopAt+{stop_pos}bp"
                            except:
                                pass

                            intron_ret_str = f"IntronRetention(+{intron_aa}aa,+{distance_to_prev}bp,frameshift={intron_frame}{stop_annotation})"

                        # Exon skipping
                        if acceptor_index > 1:
                            exon_skip_distance = abs(ann_acpt_ss[0] - dist_ann_all[2][acceptor_index - 2])
                            exon_skip_frame = exon_skip_distance % 3
                            exon_skip_aa = exon_skip_distance // 3

                            # Check for stop codons in exon skipping case (deletion)
                            has_stop_exon = False
                            if exon_skip_frame == 0 and exon_skip_distance >= 3:
                                try:
                                    skip_abs_pos = record.pos + dist_ann_all[2][acceptor_index - 2]
                                    acceptor_abs_pos = record.pos + ann_acpt_ss[0]
                                    start_pos = min(acceptor_abs_pos, skip_abs_pos)
                                    end_pos = max(acceptor_abs_pos, skip_abs_pos)

                                    sequence = get_subsequence(start_pos, end_pos)
                                    if sequence and strands[i] == '-':
                                        sequence = reverse_complement(sequence)
                                    if sequence:
                                        has_stop_exon = has_stop_codon(sequence)
                                except:
                                    pass

                            if exon_skip_frame == 0:
                                if has_stop_exon:
                                    exon_skip_str = f"ExonSkip(inframe,-{exon_skip_aa}aa,-{exon_skip_distance}bp,STOPRemoved)"
                                else:
                                    exon_skip_str = f"ExonSkip(inframe,-{exon_skip_aa}aa,-{exon_skip_distance}bp,NoStopChange)"
                            else:
                                # Frameshift: check for stop codon within context window
                                stop_annotation = ''
                                try:
                                    # Extract sequence from acceptor to end of context window
                                    start_pos = record.pos + ann_acpt_ss[0]
                                    end_pos = record.pos + cov_half
                                    sequence = get_subsequence(start_pos, end_pos)

                                    if sequence:
                                        if strands[i] == '-':
                                            sequence = reverse_complement(sequence)

                                        stop_pos = find_stop_in_frame(sequence, exon_skip_frame)
                                        if stop_pos is not None:
                                            stop_annotation = f",StopAt+{stop_pos}bp"
                                except:
                                    pass

                                exon_skip_str = f"ExonSkip(-{exon_skip_aa}aa,-{exon_skip_distance}bp,frameshift={exon_skip_frame}{stop_annotation})"

                            aa_change_acceptor = f"{intron_ret_str},{exon_skip_str}"
                            # Use intron retention frame for FRAME_CHANGE field
                            acceptor_frame_change = intron_frame
                        else:
                            aa_change_acceptor = intron_ret_str
                            acceptor_frame_change = intron_frame

            # Check for pseudoexon creation (any donor gain + acceptor gain pair)
            pseudoexon_annotation = ''
            pseudoexon_candidates = []

            # Extract donor and acceptor gains (alt > 0.5 NOT in MANE annotations)
            # Note: donor_alt_mask and acceptor_alt_mask already filter for alt > 0.5
            donor_positions = np.flatnonzero(donor_alt_mask) - cov_half
            acceptor_positions = np.flatnonzero(acceptor_alt_mask) - cov_half

            # Filter out MANE annotated sites using list comprehension (faster than loop+append)
            donor_gains = [int(p) for p in donor_positions if p not in dist_ann_set]
            acceptor_gains = [int(p) for p in acceptor_positions if p not in dist_ann_set]

            # Early exit if no gains (optimization: skip processing if no pseudoexon possible)
            if donor_gains and acceptor_gains:
                # Combine and sort all sites (using list comprehension for speed)
                all_sites = [(p, 'D') for p in donor_gains] + [(p, 'A') for p in acceptor_gains]
                all_sites.sort(key=lambda x: x[0])

                # Find consecutive donor->acceptor pairs in the sorted sequence
                for i in range(len(all_sites) - 1):
                    curr_pos, curr_type = all_sites[i]
                    next_pos, next_type = all_sites[i + 1]

                    # Valid pseudoexon: donor followed by acceptor
                    if curr_type == 'D' and next_type == 'A':
                        pseudoexon_size = next_pos - curr_pos  # No need for abs() since sorted
                        if pseudoexon_size > 0:
                            pseudoexon_candidates.append((curr_pos, next_pos, pseudoexon_size))

            # If we found pseudoexon candidates, annotate the smallest one by size (not position)
            if pseudoexon_candidates:
                # Sort by pseudoexon size (x[2]) and take the smallest pseudoexon
                pseudoexon_candidates.sort(key=lambda x: x[2])
                donor_pos, acceptor_pos, pseudoexon_size = pseudoexon_candidates[0]

                pseudoexon_frame = pseudoexon_size % 3
                pseudoexon_aa = pseudoexon_size // 3

                # Check for stop codons in pseudoexon
                has_stop_pseudoexon = False
                if pseudoexon_frame == 0 and pseudoexon_size >= 3:
                    try:
                        if strands[i] == '+':
                            start_pos = record.pos + donor_pos
                            end_pos = record.pos + acceptor_pos
                            sequence = get_subsequence(start_pos, end_pos)
                            if sequence:
                                has_stop_pseudoexon = has_stop_codon(sequence)
                        else:
                            start_pos = record.pos + acceptor_pos
                            end_pos = record.pos + donor_pos
                            sequence = get_subsequence(start_pos, end_pos)
                            if sequence:
                                sequence = reverse_complement(sequence)
                                has_stop_pseudoexon = has_stop_codon(sequence)
                    except:
                        pass

                if pseudoexon_frame == 0:
                    if has_stop_pseudoexon:
                        pseudoexon_annotation = f",PSEUDOEXON(inframe,+{pseudoexon_aa}aa,+{pseudoexon_size}bp,+STOP)"
                    else:
                        pseudoexon_annotation = f",PSEUDOEXON(inframe,+{pseudoexon_aa}aa,+{pseudoexon_size}bp,NoStop)"
                else:
                    # Frameshift: check for stop codon within context window
                    stop_annotation = ''
                    try:
                        if strands[i] == '+':
                            # Extract sequence from acceptor (end of pseudoexon) to end of context
                            start_pos = record.pos + acceptor_pos
                            end_pos = record.pos + cov_half
                            sequence = get_subsequence(start_pos, end_pos)
                            if sequence:
                                stop_pos = find_stop_in_frame(sequence, pseudoexon_frame)
                                if stop_pos is not None:
                                    stop_annotation = f",StopAt+{stop_pos}bp"
                        else:
                            # Negative strand: extract from donor to end of context
                            start_pos = record.pos + donor_pos
                            end_pos = record.pos + cov_half
                            sequence = get_subsequence(start_pos, end_pos)
                            if sequence:
                                sequence = reverse_complement(sequence)
                                stop_pos = find_stop_in_frame(sequence, pseudoexon_frame)
                                if stop_pos is not None:
                                    stop_annotation = f",StopAt+{stop_pos}bp"
                    except:
                        pass

                    pseudoexon_annotation = f",PSEUDOEXON(+{pseudoexon_aa}aa,+{pseudoexon_size}bp,frameshift={pseudoexon_frame}{stop_annotation})"

            # Format output strings - wrap each site in parentheses instead of using semicolons
            mane_donor_str = ''.join(f"({s})" for s in mane_parts['Donor']) if mane_parts['Donor'] else '.'
            mane_acceptor_str = ''.join(f"({s})" for s in mane_parts['Acceptor']) if mane_parts['Acceptor'] else '.'
            sites_gt05_donor_str = ''.join(f"({s})" for s in sites_gt05_parts['Donor']) if sites_gt05_parts['Donor'] else '.'
            sites_gt05_acceptor_str = ''.join(f"({s})" for s in sites_gt05_parts['Acceptor']) if sites_gt05_parts['Acceptor'] else '.'

            # Combined splice change - use comma instead of semicolon
            splice_change_combined = f"{splice_change_donor},{splice_change_acceptor}"

            # Combined frame change - use comma instead of semicolon
            frame_change_combined = f"{donor_frame_change},{acceptor_frame_change}"

            # Combined amino acid change - use comma instead of semicolon
            aa_change_combined = f"{aa_change_donor},{aa_change_acceptor}{pseudoexon_annotation}"

            # Cache array values for final output to avoid repeated indexing
            y_alt_pa = y[1, idx_pa, 1]
            y_ref_pa = y[0, idx_pa, 1]
            y_alt_na = y[1, idx_na, 1]
            y_ref_na = y[0, idx_na, 1]
            y_alt_pd = y[1, idx_pd, 2]
            y_ref_pd = y[0, idx_pd, 2]
            y_alt_nd = y[1, idx_nd, 2]
            y_ref_nd = y[0, idx_nd, 2]
            # Create a format string with the desired precision
            format_str = "{{}}|{{}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{:.{}f}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{}}|{{}}|{{}}|{{}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{}}|{{}}|{{}}|{{}}|{{}}|{{}}|{{}}".format(
                precision, precision, precision, precision, precision, precision, precision, precision)

            # Write delta scores for given alternative allele, gene, and calculated indices
            delta_scores.append(format_str.format(
                record.alts[i],
                genes[i],
                (y_alt_pa - y_ref_pa) * (1 - mask_pa),
                (y_ref_na - y_alt_na) * (1 - mask_na),
                (y_alt_pd - y_ref_pd) * (1 - mask_pd),
                (y_ref_nd - y_alt_nd) * (1 - mask_nd),
                (y_alt_pa - y_ref_pa),
                (y_ref_na - y_alt_na),
                (y_alt_pd - y_ref_pd),
                (y_ref_nd - y_alt_nd),
                idx_pa - cov_half,
                idx_na - cov_half,
                idx_pd - cov_half,
                idx_nd - cov_half,
                y_alt_pa,
                y_alt_na,
                y_alt_pd,
                y_alt_nd,
                mane_donor_str,
                mane_acceptor_str,
                sites_gt05_donor_str,
                sites_gt05_acceptor_str,
                splice_change_combined,
                frame_change_combined,
                aa_change_combined))

    return delta_scores 