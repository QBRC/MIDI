from utils.process_data import *
from utils.smile_rel_dist_interpreter import *
from base_line_models import *
from drug_transformer import *
import scipy.stats
from sklearn.mixture import BayesianGaussianMixture
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter
#import keras_nlp
from tensorflow.keras import initializers
import json
tf.keras.utils.set_random_seed(812)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
from random import seed
from random import sample

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
import selfies as sf
import numpy as np
import Geneformer.geneformer as ge
import gseapy as gp
import sys 


def filtering_raw_gene_expression(gene_expression: pd.DataFrame)->pd.DataFrame:
    """
    Compute the variance of each gene expression, and also return 
    the zero amount of gene expression
    
    Parameters:
    -----------
    gene_expression: cell line gene expression input
    
    Returns:
    --------
    dataframe with gene expression variance and zero amount
    """
    std_list = []
    zeros_list = []
    filtered_list = []
    #filtered_index_list = [] 
    std_threshold = 1
    zero_threshold = 250
    gene_names = gene_expression.columns
    index = 0
    for i in gene_names:
        #print(index)
        std = np.nanstd(gene_expression[i])
        std_list.append(std)
        zeros_num = list(gene_expression[i]).count(0)
        zeros_list.append(zeros_num)
        if std < std_threshold or zeros_num > zero_threshold:
            #gene_expression = gene_expression.drop([i],axis=1)
            filtered_list.append(i)
            #filtered_index_list.append(index)
            #print("im here in condition")
        #print(index)
    index+= 1
    gene_expression = gene_expression.drop(filtered_list,axis=1)
    
    return gene_expression

def smile_cl_converter(smile):
    new_smile = ''
    for i in range(len(smile)):
        if smile[i] == 'C':
            if not i == len(smile) - 1:
                if smile[i+1] == 'l':
                    new_smile += 'L'
                else:
                    new_smile += smile[i]
            else:
                new_smile += smile[i]
        elif smile[i] == 'l' and smile[i-1] == 'C':
            continue

        elif smile[i] == 'B':
            if not i == len(smile) - 1:
                if smile[i+1] == 'r':
                    new_smile += 'B'
                else:
                    return None
                    #new_smile += smile[i]
        elif smile[i] == 'r' and smile[i-1] == 'B':
            continue
            
        else:
            #new_smile.append(smile[i])
            new_smile+=smile[i]
    return new_smile

def generate_interpret_smile(smile):
    new_smile = smile_cl_converter(smile)
    length = len(new_smile)
    rel_distance_whole = []
    interpret_smile_whole = []
    projection_whole = []

    for i in range(length):
        symbol, dist = symbol_converter(new_smile[i])
        if symbol == None:
            continue
        elif symbol not in vocabulary_drug:
            return None
        else:
            rel_distance, interpret_smile, projection = smile_rel_dis_interpreter(new_smile, i)
            #length_seq = len(rel_distance)
            rel_distance_whole.append(rel_distance)
            interpret_smile_whole.append(interpret_smile)
            projection_whole.append(projection)

    rel_distance_whole = np.stack(rel_distance_whole)
    projection_whole = np.stack(projection_whole)
    interpret_smile_whole = np.stack(interpret_smile_whole)

    return interpret_smile_whole


def extract_atoms_bonds(weight_min_max,smile):
    mol = Chem.MolFromSmiles(smile)
    resolution = (weight_min_max.max()-weight_min_max.min())/40
    resolution_color = 1/40
    highlight_atoms = []
    weight_atoms_indices = list(np.argsort(-weight_min_max.diagonal())[0:5])
    weight_atoms_indices = [int(kk) for kk in weight_atoms_indices]
    colors = {}
    value_color_list = []
    for h in weight_atoms_indices:
        value_color = ((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        #colors[h] = ( 1, 1-value_color, 1-value_color)
        value_color_list.append(1-value_color)
    max_value_color = np.array(value_color_list).max()
    min_value_color = np.array(value_color_list).min()
    range_value_color = max_value_color - min_value_color
    """
    for h in weight_atoms_indices:
        value_color = 1-((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        new_value_color = ((value_color - min_value_color)/range_value_color)*(0.7)
        #colors[h] = ( 1, new_value_color, new_value_color)
        colors[h] = ( 1, 0, 0)
    """
    highlight_bond = []
    highlight_bond_atoms = []
    weight_bond = []
    colors_bond = {}
    bond_idx_ = []
    value_color_list_bond = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mid_weight = weight_min_max[bond_i,bond_j]#(weight_min_max[bond_i] + weight_min_max[bond_j]) / 2
        weight_bond.append(mid_weight)
        bond_idx_.append(bond_idx)
    highlight_indices = list(np.argsort(-np.array(weight_bond)))[0:7]#[0:5]

    for bond_idx, bond in enumerate(mol.GetBonds()):
        if bond_idx in highlight_indices:
            bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_i_weight = weight_min_max.diagonal()[bond_i]
            bond_j_weight = weight_min_max.diagonal()[bond_j]
            highlight_bond_atoms.append(bond_i)
            highlight_bond_atoms.append(bond_j)
            """
            if bond_i_weight > bond_j_weight:
                highlight_bond_atoms.append(bond_i)
            else:
                highlight_bond_atoms.append(bond_j)
            """
    weight_atoms_indices = weight_atoms_indices + highlight_bond_atoms

    for h in weight_atoms_indices:
        value_color = 1-((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        new_value_color = ((value_color - min_value_color)/range_value_color)*(0.7)
        #colors[h] = ( 1, new_value_color, new_value_color)
        colors[h] = ( 1, 0, 0)
    
    for bond_idx, bond in enumerate(mol.GetBonds()):
        bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #mid_weight = weight_min_max[bond_i,bond_j]#(weight_min_max[bond_i] + weight_min_max[bond_j]) / 2
        mid_weight = weight_bond[bond_idx]
        #weight_bond.append(mid_weight)
        #if bond_i in weight_atoms_indices:
        if bond_idx in highlight_indices:
            highlight_bond.append(bond_idx)
            value_color_ = ((mid_weight-weight_min_max.min())/resolution)*resolution_color
            #value_color_list_bond.append(1-value_color_)
            colors_bond[bond_idx] = (1, 1-value_color_, 1-value_color_)
    
    return weight_atoms_indices, highlight_bond, colors, colors_bond

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol



if __name__ == '__main__':
	"""
	Load model
	"""
	if len(sys.argv) > 2:
	    drug_smile_input = sys.argv[2]
        gene_profile = sys.argv[3]
	    job_id = sys.argv[1]
	    print(f"Smile Sequence input is, {drug_smile_input}")
	else:
	    print("No Smile Input")
	    sys.exit(0)

    gene_profile_ = pd.read_csv(gene_profile)
	#ensemble_id = pyreadr.read_r('Ling-Tingyi/LCCL_input/RNA-CCLE_RNAseq.annot.rds')[None]

	#k = drug_transformer_(gene_embeddings)
    
	
    with open('gene_embedding_important.npy', 'rb') as f:
        gene_embeddings = np.load(f)

    gene_name_avail_geneformer = list(np.load('gene_names.npy'))

    k = drug_transformer_(gene_embeddings)#, relative_pos_enc_lookup=relative_pos_embedding)
    model_midi = k.model_construction_midi(if_mutation=True)
    model_midi.load_weights('Pre_train_model/midi_55_epochs_prior_3000_pairs_with_drug_regularizer_softmax_temperature_9_training.h5')

    """
    extract self-attention score for drug structure, and cross-attention score
    for gene ranking
    """
    feature_select_score_model_drug = att_score_self_enco(model_midi,7)
    feature_select_score_model_gene = att_score_self_enco(model_midi,31)

    gene_expression = np.array(gene_profile)[0]
    gene_mutation = np.array(gene_profile)[1]

    df_data = pd.read_csv(os.path.join(midi_path,'train_data_midi.csv'))
    df_data.set_index("drug_name", inplace=True)


	drug_name = drug_names[21]
	#print(drug_name)
	df_drug_train_data = df_data.loc[drug_name]
	#batch_smile_seq = df_drug_train_data['smile_seq'] 
	batch_smile_seq = [drug_smile_input]
	#batch_interpret_smile = df_drug_train_data['interpret_smile']
	#batch_interpret_smile = [interpret_drug_smile_input]
	batch_cell_line_name = [df_drug_train_data['cell_line_name'][0]]
	batch_drug_response = [df_drug_train_data['drug_response'][0]]
	batch_drug_name = [drug_name for i in range(len(batch_smile_seq))]
	drug_atom_one_hot_chunk, drug_rel_position_chunk, edge_type_matrix_chunk,\
	drug_smile_length_chunk, gene_expression_bin_chunk, gene_mutation_bin_chunk, gene_prior_chunk = \
	extract_input_data_midi(batch_drug_name, batch_smile_seq, batch_cell_line_name, batch_drug_response,\
                            gene_expression, gene_mutation)
	    
	batch_shape = drug_atom_one_hot_chunk.shape[0]
	mask = tf.range(start=0, limit=100, dtype=tf.float32)
	mask = tf.broadcast_to(tf.expand_dims(mask,axis=0),shape=[batch_shape,100])
	mask = tf.reshape(mask, shape=(batch_shape*100))
	mask = mask < tf.cast(tf.repeat(drug_smile_length_chunk,repeats=100),tf.float32)
	mask = tf.where(mask,1,0)
	mask = tf.reshape(mask, shape=(batch_shape,100))
	mask = tf.expand_dims(mask, axis=-1)

    prediction, score_cross_global, X_global, Y_gene, \
            Y_gene_embedding, X_global_, att_score_global2, Y_global = \
            model_midi((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
                        drug_smile_length_chunk, drug_rel_position_chunk, \
                        edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))

	feature_select_score_drug = feature_select_score_model_drug.predict((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
	                                                                drug_smile_length_chunk, drug_rel_position_chunk, \
	                                                                edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))[1]

    feature_select_score_gene = feature_select_score_model_gene.predict((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
                                                                    drug_smile_length_chunk, drug_rel_position_chunk, \
                                                                    edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))[1][:,0,:]

	feature_select_score_drug_whole.append(feature_select_score_drug[0])
	#drug_feature_select_score.append(feature_select_score_drug)

	#feature_select_score = drug_feature_select_score[4]

    df_drug_eff = pd.DataFrame(list(zip(['negative_activity_area'], [np.array(prediction)[0][0]])),
                             columns=['measurement', 'value'])

    df_drug_eff.to_csv('output/drug_effect_prediction.csv')


    top_genes_score, top_genes_index = tf.math.top_k(feature_select_score_gene[5], k=6144)
    top_gene_names = np.array([gene_name_avail_geneformer[j] for j in top_genes_index])

    """
    Create a dataframe for the gene ranking list
    """
    df_rank_gene = pd.DataFrame(list(zip(list(top_gene_names), list(np.array(top_genes_score)))),
                                 columns=['gene_name', 'attention_score'])

    df_rank_gene.to_csv('output/gene_rank.csv')

    #total_top_gene_rank.set_index('drug_names',inplace=True)
    #x = total_top_gene_rank.loc[drug_name_plot]['gene_name']
    sns.set_style("white")
    x_ = list(range(6144))
    y = list(top_genes_score)

    plt.figure(figsize=(5,5))
    plt.plot(x_,y, 'o',markersize=5)

    plt.savefig('output/gene_attention_scores.png',dpi=300)

    plt.cla()

	plt.figure()
    g = sns.heatmap(feature_select_score_drug[0][0:drug_smile_length_chunk[0],0:drug_smile_length_chunk[0]], cmap="Blues")
    sns.set(rc={"figure.figsize":(10,10)})
    interpret_smile = list(generate_interpret_smile(batch_smile_seq[0])[0])
    x_labels = list(interpret_smile)
    y_labels = list(interpret_smile)
    #atoms_drug.append(x_labels)
    #y_labels.reverse()
    g.set_xticks(range(len(x_labels)), labels=x_labels)
    #g.set_yticks(range(1), labels=[' '])
    g.set_yticks(range(len(y_labels)), labels=y_labels)
    g.tick_params(axis='x', rotation=0)
    g.set(title = drug_smile_input)
    g.plot()
    figure = g.get_figure()
    figure.savefig('output/heatmap.png', dpi=300)
    
    from PIL import Image
    from io import BytesIO
    weight_atoms_indices, highlight_bond, colors, colors_bond = extract_atoms_bonds(feature_select_score_drug[0], batch_smile_seq[0])
    #weight_atoms_indices, highlight_bond, colors, colors_bond = extract_atoms_bonds(feature_select_score_drug_whole[0], batch_smile_seq[0])
    mol = Chem.MolFromSmiles(batch_smile_seq[0])
    mol = mol_with_atom_index(mol)
    d2d = rdMolDraw2D.MolDraw2DCairo(500,300)
    option = d2d.drawOptions()
    option.legendFontSize = 18
    option.bondLineWidth = 1.5
    option.highlightBondWidthMultiplier = 20
    option.updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
    d2d.DrawMolecule(mol,highlightAtoms=weight_atoms_indices,highlightAtomColors=colors)#, highlightBonds=highlight_bond, highlightBondColors=colors_bond)
    #2d.DrawMolecule(mol,highlightAtoms=weight_atoms_indices, highlightBonds=highlight_bond, highlightBondColors=colors_bond)
    #d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    Image.open(bio)
    d2d.FinishDrawing()
    d2d.WriteDrawingText("output/molecule.png")

    