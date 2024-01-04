import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# installed scikit-learn
from sklearn.model_selection import train_test_split
import statistics
from matplotlib.backends.backend_pdf import PdfPages

device = torch.device("cpu")
#device = torch.device("gpu")

# device = torch.device("cuda:0") # Uncomment this to run on GPU

class Model(nn.Module):
    def __init__(self, in_features=40, h1=20, h2=5, out_feature=1, use_embedding=False, embedding_dim=5, embedding_dict_size=21 ):
        super().__init__()

        self.use_embedding = use_embedding
        self.in_features = in_features
        self.embedding_dim = embedding_dim

        # TODO finish
        # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        if use_embedding:
            # self.emb = nn.Embedding( embedding_dict_size,embedding_dim, padding_idx=4 )
            self.emb = nn.Embedding( embedding_dict_size,embedding_dim )
            self.fc1 = nn.Linear( in_features*embedding_dim, h1 )

        else:
            self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear( h1, h2 )
        self.out = nn.Linear( h2, out_feature )

    def forward(self, x):
        if self.use_embedding:
            x = self.emb(x).view(-1, self.embedding_dim*self.in_features )
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        # x = F.relu(self.out(x))
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MyTimer:
    def __init__( self ):
        self.start = time.perf_counter()
    def time( self ):
        self.current_time = time.perf_counter() - self.start
        return self.current_time

def train_and_store( seed_par, seed_repeat, input_size, h1_size, h2_size, learning_rate, plot_bool,
                     t_max_time, epochs_modulo, t_patience, lr_to_add_to_min_lr, b_do_early_stop, use_embedding,
                     embedding_dim, embedding_dict_size ):

    # TODO make a dictionary
    list_seed_repeat = []
    list_h1_size = []
    list_h2_size = []
    list_learning_rate = []
    list_learning_rate_up = []
    list_pearson_corr = []
    list_mean_abs_delt_iRT = []
    list_rel_mean_abs_delt_iRT = []
    list_b_early_stop = []
    list_b_max_time = []
    list_time_used = []
    list_use_embedding = []
    list_embedding_dim = []

    for c_seed_repeat in range( 0, seed_repeat ):

        print('H1: ' + str(h1_size))
        print('H2: ' + str(h2_size))
        print('learning rate: ' + str(learning_rate) )
        print('learning rate up: ' + str(lr_to_add_to_min_lr) )
        print('seed repeat: ' + str( c_seed_repeat + 1 ) )

        # special stop conditions

        b_early_stop = False
        b_max_time = False

        # make model, input size is dynamic based on feat
        torch.manual_seed( seed_par*c_seed_repeat )

        model = Model(in_features=input_size, h1=h1_size, h2=h2_size, out_feature=1, use_embedding=use_embedding,
                      embedding_dim=embedding_dim, embedding_dict_size=embedding_dict_size )

        # if use_embedding:
        #     # model = Model( in_features=input_size, h1=h1_size, h2=h2_size, out_feature=1, use_embedding=use_embedding,
        #     #                embedding_dim=embedding_dim )
        #     model = Model(in_features=input_size, h1=h1_size, h2=h2_size, out_feature=1, use_embedding=use_embedding,
        #                   embedding_dim=embedding_dim )
        # else:
        #     model = Model( in_features=input_size, h1=h1_size, h2=h2_size, out_feature=1 )

        # define loss and optimizer
        criterion = nn.MSELoss()
        # criterion = nn.MSELoss( reduction='mean' )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # margin 0 should correspond to a fix learning rate
        min_lr = learning_rate
        max_lr = learning_rate + lr_to_add_to_min_lr
        sched_step_size_up = 50

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, cycle_momentum=False,
                                                      mode='triangular2', step_size_up=sched_step_size_up)

        early_stopper = EarlyStopper(patience=t_patience, min_delta=0)
        my_timer = MyTimer()

        # learning curves
        l_plot_epochs = []
        l_plot_train_loss = []
        l_plot_test_loss = []
        l_lr_rate = []
        epochs_early_stop_min = 0

        # data_loader = DataLoader(X_train, batch_size=1)

        for i in range(epochs):

            l_plot_epochs.append(i)
            # go forward and predict
            # y_pred = model.forward(X_train)

            # y_pred = model(X_train).flatten()

            # new code to test
            y_pred = model(X_train)

            # print(y_pred)
            # print(y_train)

            loss = criterion(y_pred, y_train)
            loss_number = loss.detach().numpy()
            l_plot_train_loss.append(loss_number)

            # clear the gradients
            optimizer.zero_grad()
            loss.backward()
            # update model weights
            optimizer.step()
            scheduler.step()

            # store learning rate schedule for plotting
            # l_lr_rate.append(optimizer.state_dict()['param_groups'][0]['lr'])
            # c_learning_rate = scheduler.get_lr()
            c_learning_rate = scheduler.get_last_lr()
            c_learning_rate = c_learning_rate[0]
            l_lr_rate.append(c_learning_rate)
            # print(float(c_learning_rate))

            # early stop
            # calculate test loss
            y_pred_test = model(X_test)
            loss_test = criterion(y_pred_test, y_test)
            loss_number_test = loss_test.detach().numpy()
            l_plot_test_loss.append(loss_number_test)
            if b_do_early_stop:
                if early_stopper.early_stop(loss_number_test):
                    print(f'EARLY STOP: Epoch: {i:4}, loss: {loss_number:6.4}, test loss: {loss_number_test:6.4}')
                    b_early_stop = True
                    epochs_early_stop_min = i - t_patience
                    break

            if i % epochs_modulo == 0:
                t_time_reached = my_timer.time()
                if t_time_reached > t_max_time:
                    print(f'Time reached:  {t_time_reached:05.1f}s, max time allowed: {t_max_time}s')
                    b_max_time = True
                    break

            if i % ( epochs / 5 ) == 0:
                print(f'Epoch: {i:4}, loss: {loss_number:08.6f}, test loss: {loss_number_test:08.6f}, lr: {c_learning_rate:08.6f}')


        y_pred_test = model(X_test)

        # convert variable between 0 and 1 to iRT
        px = var_to_iRT(y_test.flatten().detach().numpy())
        py = var_to_iRT(y_pred_test.flatten().detach().numpy())

        # pearson correlation
        pears_corr = np.corrcoef(px, py)[0][1]
        print('pearson correlation test data set: ' + str(pears_corr))

        # abs delta iRT
        abs_delt_iRT = abs(px - py)
        mean_abs_delt_iRT = statistics.mean(abs_delt_iRT)
        print('mean absolute delta iRT: ' + str(mean_abs_delt_iRT))

        # rel mean abs delta iRT
        rel_mean_abs_delta_iRT = mean_abs_delt_iRT / (iRT_max - iRT_min) * 100
        print('relative mean absolute delta iRT: ' + str(rel_mean_abs_delta_iRT))

        # TODO plot rel_mean_abs_delta_iRT, time out
        if plot_bool:
            # plot scatter plot on test after training is finished
            # plot to pdf
            np.random.seed = 10
            fig = plt.figure()
            idx_sample = np.random.choice( range(len(px)), len(px) if len(px) < 1000 else 1000, replace=False)
            spx = px[idx_sample]
            spy = py[idx_sample]
            plt.scatter(spx, spy, color='red', alpha=0.1)
            text_x = min(spx) + ( max(spx) - min(spx) ) / 20
            text_y = max(spy) - ( max(spy) - min(spy) ) / 10
            # print(rel_mean_abs_delta_iRT)
            # print( f'RMAD iRT: {rel_mean_abs_delta_iRT:07:5f}' )
            plt.text( text_x, text_y, f'RMAD iRT: {rel_mean_abs_delta_iRT:7.5f}' )
            plt.title('Test Data Set: H1 ' + str(h1_size) + ', H2 ' + str(h2_size) + ', LR ' + str(
                learning_rate) + ', LRup ' + str(lr_to_add_to_min_lr) + ', R ' + str(c_seed_repeat) )
            plt.xlabel('empirical iRT')
            plt.ylabel('predicted iRT')
            my_pdf.savefig(fig)
            plt.close()

            # line plot of the train and test loss
            fig2 = plt.figure()
            text_x = max(l_plot_epochs) - ( max(l_plot_epochs) - min(l_plot_epochs) ) / 3
            y_losses = l_plot_train_loss + l_plot_test_loss
            text_y = max(y_losses) - ( max(y_losses) - min(y_losses) ) / 2
            plt.text( text_x, text_y, f'RMAD iRT: {rel_mean_abs_delta_iRT:7.5f}' )
            plt.plot( l_plot_epochs, l_plot_train_loss, label="train loss", color='grey' )
            plt.plot( l_plot_epochs, l_plot_test_loss, label="test loss", color='orange' )
            if b_early_stop:
                plt.axvline( x=epochs_early_stop_min, color='red', linestyle='-', label='early stop' )
            plt.title('Losses: H1 ' + str(h1_size) + ', H2 ' + str(h2_size) + ', LR ' + str(
                learning_rate) + ', LRup ' + str(lr_to_add_to_min_lr) + ', R ' + str(c_seed_repeat) )
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            my_pdf.savefig(fig2)
            plt.close()

            # line plot of the learning rate
            # TODO label the learning rate parameters better
            fig3 = plt.figure()
            plt.plot( l_plot_epochs, l_lr_rate, label="learning rate", color='grey' )
            text_x = max(l_plot_epochs) - ( max(l_plot_epochs) - min(l_plot_epochs) ) / 3
            text_y = max(l_lr_rate) - ( max(l_lr_rate) - min(l_lr_rate) ) / 2
            plt.text( text_x, text_y, f'RMAD iRT: {rel_mean_abs_delta_iRT:7.5f}' )
            plt.title('Learning rate: H1 ' + str(h1_size) + ', H2 ' + str(h2_size) + ', LR ' + str(
                learning_rate) + ', LRup ' + str(lr_to_add_to_min_lr) + ', R ' + str(c_seed_repeat) )
            plt.xlabel('epochs')
            plt.ylabel('learning rate')
            plt.legend()
            my_pdf.savefig(fig3)
            plt.close()

            # plt.scatter(px, py)
            # plt.title('Test Data Set')
            # plt.ylabel('predicted iRT')
            # plt.xlabel('empirical iRT')
            # # plt.ion()
            # plt.show(block=False)
            # plt.pause(3)
            # # plt.ioff()
            # plt.close()


        # save results
        list_seed_repeat.append(c_seed_repeat)
        list_h1_size.append(h1_size)
        list_h2_size.append(h2_size)
        list_learning_rate.append(learning_rate)
        list_learning_rate_up.append((lr_to_add_to_min_lr))
        list_pearson_corr.append(pears_corr)
        list_mean_abs_delt_iRT.append(mean_abs_delt_iRT)
        list_rel_mean_abs_delt_iRT.append(rel_mean_abs_delta_iRT)
        list_b_early_stop.append(b_early_stop)
        list_b_max_time.append(b_max_time)
        list_time_used.append(my_timer.time())
        list_use_embedding.append(use_embedding)
        list_embedding_dim.append((embedding_dim))

        print()

    data_dict = { 'seed_repeat': list_seed_repeat, 'h1_size': list_h1_size, 'h2_size': list_h2_size,
                  'learning_rate': list_learning_rate, 'learning_rate_up': list_learning_rate_up,
                  'pearson_correlation': list_pearson_corr,
                  'mean_abs_delt_iRT': list_mean_abs_delt_iRT, 'rel_mean_abs_delt_iRT': list_rel_mean_abs_delt_iRT,
                  'early_stop': list_b_early_stop, 'max_time': list_b_max_time, 'time_used': list_time_used,
                  'use embedding': list_use_embedding, 'embedding_dim': list_embedding_dim }
    return data_dict

def parse_library(url, t_sep, do_print, do_remove_modified_pep, do_remove_duplicate_rows):
    df = pd.read_csv(url, sep=t_sep)
    if do_print: print('parsed rows: ', len(df.index))

    df['pep'] = df['r_fileName'] + df['eg_primaryIsotopeCanonicalMoleculeId']
    df = df[['pep', 'eg_iRT', 'eg_primaryIsotopeCanonicalMoleculeId', 'eg_aminoAcidSequence', 'tg_Q1']]

    if do_remove_modified_pep:
        # removing modified peptides
        if do_print: print('removing modified peptides...')
        t_sel = df['eg_primaryIsotopeCanonicalMoleculeId'].str.contains('\[')
        df = df[~t_sel]

    if do_remove_duplicate_rows:
        # select unique rows
        # TODO remove duplicates for run x modpep (without the rest)
        if do_print: print('removing duplicates...')
        df = df[~df.duplicated()]
        if do_print: print('unique rows: ' + str(len(df.index)))

    if do_print: print()

    return df


def generate_simple_features(df, do_remove_col_with_nan, do_normalize, do_print):
    # generate features to learn
    # 20 proteinogenic amino acids
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for aa in aas:
        df[aa] = df['eg_aminoAcidSequence'].str.count(aa)
        df[aa] = df[aa].astype(float)

    feat = aas.copy()

    # last amino acid feature scores
    # TODO use generic 'score_' names
    t_str = 'last'
    for aa in aas:
        df[t_str + aa] = [tmp[-1] for tmp in df['eg_aminoAcidSequence']]
        df[t_str + aa] = df[t_str + aa].str.count(aa)
        df[t_str + aa] = df[t_str + aa].astype(float)
        feat.append(t_str + aa)

    df['len'] = df['eg_aminoAcidSequence'].str.len()
    df['len'] = df['len'].astype(float)
    feat.append('len')

    # Q1 doesn't work, would have to use mass instead
    # df['tg_Q1'] = df['tg_Q1'].astype(float)
    # feat.append('tg_Q1')

    # removing columns with NaN
    if do_remove_col_with_nan:
        t_len = len(feat)
        t_l = []
        for idx in range(t_len):
            if df[feat[idx]].isnull().any():
                t_l.append(idx)
                if do_print: print('removing column with NaN: ' + feat[idx])

        for idx in t_l[::-1]:
            feat.pop(idx)

    if do_normalize:
        if do_print: print('normalizing features')
        t_global_min = df[feat].min().min()
        t_global_max = df[feat].max().max()
        for col in feat:
            df[col] = (df[col] - t_global_min) / (t_global_max - t_global_min)

    # generic name
    df['target_float'] = df['eg_iRT']

    return (df, feat)

def generate_features_with_sequence_embedding( df, do_print ):

    feat = []

    # determine maximal aa length
    df['len'] = df['eg_aminoAcidSequence'].str.len()

    t_max_aa_length = max(df['len'])
    if do_print: print('maximal pep sequence length: ' + str(t_max_aa_length))

    l_str = []
    for row_idx in df.index:
        t_fill = t_max_aa_length - df['len'][row_idx]
        t_fill_str = 'X'*t_fill
        l_str.append( df['eg_aminoAcidSequence'][row_idx] + t_fill_str )
    if do_print: print( 'filling aa sequence with X to max size' )

    dict_columns = {}
    # initialize the filler X as zero
    dict_char_emb = {'X': 0}
    char_count = 1
    for st in l_str:
        i = 0
        for char in st:
            if char not in dict_char_emb.keys():
                dict_char_emb[char] = char_count
                char_count += 1
            if i not in dict_columns:
                dict_columns[i] = [ dict_char_emb[char] ]
            else:
                dict_columns[i].append( dict_char_emb[char] )
            i += 1
    if do_print: print( 'X represented as: ' + str( dict_char_emb['X']) )
    emb_dict_size = len(dict_char_emb.keys())
    if do_print: print( 'total chars in dict including X: ' + str(emb_dict_size) )

    new_df = pd.DataFrame()
    for key in dict_columns.keys():
        c_feat = 'score_' + str(key)
        feat.append(c_feat)
        new_df[c_feat] = dict_columns[key].copy()

    # generic name
    iRT = list(df['eg_iRT'].copy( deep=True ))
    new_df['target_float'] = iRT

    return new_df, feat, emb_dict_size

def parse_from_generic_file( url, t_sep='\t', do_print=True ):

    df = pd.read_csv(url, sep=t_sep)
    if do_print: print('parsed rows: ', len(df.index))

    feat = []
    for column in df.columns:
        if str(column).startswith( 'score_' ):
            feat.append(column)

    embedding_dict = {}
    for s_column in feat:
        for entry in df[s_column]:
            if entry not in embedding_dict.keys():
                embedding_dict[entry] = 0
    embedding_dict_size = len(embedding_dict.keys())

    return df, feat, embedding_dict_size

############################################
# main script
############################################
# TODO loop through h2, activation function, num layers, embedding dim

# r in front of the string turns escaping off, -> raw string, r can be upper or lowercase
#url = r'C:\Users\lukas\PycharmProjects\iRTPredTest\148_test_rt_pred\Human - HeLa.xls'
#url = r'C:\Users\lukas\PycharmProjects\testpy\Human - HeLa.xls'
url = r"C:\Users\lukas\PycharmProjects\testpy\2024_January_04_07h52PM_features.xls"

# input parameters
seed_par = 41
# h1_size = 64
h2_size = 128
epochs = 2000
epochs_modulo = 10

# repeat each experiment so many times with different random seeds to initialize the weights
seed_repeat = 1
# maximal allowed time to train one network
# for 3^6 models (5 parameters with 3 values and 3 replicates) ~ 2 minutes per model (in 24h) -> 120s
t_max_time = 600
# patience 30 with min delta 0 works well when one epoch ~24k training data points for early stopping
b_do_early_stop = True
t_patience = 25

# otherwise hard coded library file
b_parse_from_generic_file = True

# training with embedding is much slower
use_embedding = True
embedding_dim = 8

# l_h1_size = [1024,512,256]
# l_lr = [0.0001,0.0005]
# l_lr_up = [0.005,0.001]

l_h1_size = [512]
l_lr = [0.0005]
l_lr_up = [0.001]

b_plot = True

#print( 'seed: ' + str(seed_par) )
print('epochs: ' + str(epochs) )
print('seed repeat: ' + str(seed_repeat) )
print('max time: ' + str(t_max_time) )
print('do early stop: ' + str(b_do_early_stop) )
print('patience: ' + str(t_patience) )
print('use embedding: ' + str(use_embedding) )
print('embedding dimension: ' + str(embedding_dim) )
print()

############################################
# data pre processing
############################################

df, feat, embedding_dict_size = None, None, None

if b_parse_from_generic_file:
    df, feat, embedding_dict_size = parse_from_generic_file( url )
    print('number of features: ' + str(len(feat)))
else:
    df = parse_library( url=url, t_sep='\t', do_print=True, do_remove_modified_pep=True, do_remove_duplicate_rows=True )

    # TODO test the embedding features
    if not use_embedding:
        df, feat = generate_simple_features( df=df, do_remove_col_with_nan=True, do_normalize=False, do_print=True )
    else:
        df, feat, embedding_dict_size = generate_features_with_sequence_embedding( df=df, do_print=True )
        print('embedding dict size: ' + str(embedding_dict_size))


# formatting features for use in torch
X = df[feat]

# TODO make this part more generic
# convert iRT between 0 and 1
# eg_iRT is renamed to target_float in the functions above
iRT_min = min(df['target_float'])
iRT_max = max(df['target_float'])
print( 'iRT min: ' + str(iRT_min))
print( 'iRT max: ' + str(iRT_max))

def iRT_to_var( iRT_float ):
    return ( iRT_float - iRT_min ) / ( iRT_max - iRT_min )

def var_to_iRT( iRT_var ):
    return ( iRT_var * ( iRT_max - iRT_min ) ) + iRT_min

y = iRT_to_var( df['target_float'] )

X = X.values
y = y.values

# print('X.values')
# print(type(X))
# print(X[:3])
# print()

# TODO check whether this also randomizes data points, split by
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5 )

if not use_embedding:
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
else:
    # TODO check whether torch long would be better
    X_train = torch.tensor(X_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)

# print('X_train')
# print(type(X_train))
# print(X_train[:3])
# print()

# reshape to have the same shape as the output
y_train = torch.FloatTensor(y_train).view(len(y_train), -1)
y_test = torch.FloatTensor(y_test).view(len(y_test), -1)

print( 'train size: ' + str(len(y_train)) + ', test size: ' + str(len(y_test)) )
print()

# for printing out to files
t_out_date_time_str = datetime.datetime.now().strftime("%Y_%B_%d_%Ih%M%p")

# print out processed input table with features
df.to_csv( t_out_date_time_str + '_features.xls', sep='\t' )

# plotting to pdf
my_pdf = PdfPages( t_out_date_time_str + '_hyperparsearch.pdf' )

############################################
# hyperparameter opt loops
############################################

data_dict = {}
# for h1_size in [ 64, 128, 256 ]:
for h1_size in l_h1_size:

    # for learning_rate in [ 0.0001, 0.0005, 0.001 ]:
    for learning_rate in l_lr:

        # for learning rate scheduler with triangular shape
        # if this is zero the learning rate is constant
        # for learning_rate_up in [ 0, 0.001, 0.005 ]:
        for learning_rate_up in l_lr_up:

            results_dict = train_and_store( seed_par=seed_par, seed_repeat=seed_repeat, input_size=len(feat),
                                            h1_size=h1_size, h2_size=h2_size, learning_rate=learning_rate,
                                            plot_bool=b_plot, t_max_time=t_max_time, epochs_modulo=epochs_modulo,
                                            t_patience=t_patience, lr_to_add_to_min_lr=learning_rate_up,
                                            b_do_early_stop=b_do_early_stop, use_embedding=use_embedding,
                                            embedding_dim=embedding_dim, embedding_dict_size=embedding_dict_size )

            # if first time returned and data_dict emtpy
            if not data_dict:
                data_dict = results_dict
            else:
                for ( key, value ) in results_dict.items():
                    data_dict[key] = data_dict[key] + value.copy()

            print()
            print()

############################################
# process and store results
############################################
df_results = pd.DataFrame(data_dict)

df2=df_results.groupby(['h1_size', 'h2_size', 'learning_rate'])
# TODO this throws a warning
df_results['min_rel_mean_abs_delt_iRT'] = df2['rel_mean_abs_delt_iRT'].transform(min)
df_results['mean_rel_mean_abs_delt_iRT'] = df2['rel_mean_abs_delt_iRT'].transform(statistics.mean)
# df_results['sd_rel_mean_abs_delt_iRT'] = df2['rel_mean_abs_delt_iRT'].transform(statistics.stdev)
df_results['sd_rel_mean_abs_delt_iRT'] = df2['rel_mean_abs_delt_iRT'].transform(lambda x: float('nan') if len(x) <= 1 else statistics.pstdev(x) )

df3=df_results.groupby(['h1_size', 'h2_size', 'learning_rate'])['rel_mean_abs_delt_iRT'].describe()

df_results.to_csv( t_out_date_time_str + '_hyperparsearch_raw.xls', sep='\t' )
df3.to_csv( t_out_date_time_str + '_hyperparsearch_summary_stat.xls', sep='\t' )

# close pdf
my_pdf.close()
