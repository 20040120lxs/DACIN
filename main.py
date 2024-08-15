import numpy as np
import utilss
import DACINModel
import clustering
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='dataset/letter/missing_ratio/', type=str)
    parser.add_argument('-model_path', default='./model/', type=str)
    # for imputation
    parser.add_argument('-epochs', default=1000,type=int)
    parser.add_argument('-iter_num', default=5, type=int)
    parser.add_argument('-batch_size', default=1474,type=int)
    parser.add_argument('-hint_rate', default=0.9, type=float)
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    # for clustering
    parser.add_argument('-k', default=6,type=int, help='number of clusters')
    parser.add_argument('-d', default=12,type=int, help='dimension of each subspace')
    parser.add_argument('-ro', default=0.4, type=float, help='hyperparameter')
    parser.add_argument('-alpha', default=8,type=int, help='hyperparameter')

    args = parser.parse_args()
    return args

def main(args):

    ori_data_x = np.loadtxt(args.data_dir + "data.txt", delimiter=",").astype(np.float32)
    ori_data_label = np.loadtxt(args.data_dir + "data_labels.txt", delimiter=",")

    ori_data_label = ori_data_label.reshape((-1, 1)).astype(int)

    print("ori_data_x: ", ori_data_x.shape, "ori_data_label: ", ori_data_label.shape)
    num_class = np.unique(ori_data_label).size
    print("num_class: ", num_class)

    no, dim = ori_data_x.shape
    data_x, norm_parameters = utilss.normalization(ori_data_x)
    print("norm_parameters: ", norm_parameters)

    # numerical encoding
    # data_label_numerical, norm_label_parameter = utilss.normalization(ori_data_label)

    # one-hot encoding
    data_label_one_hot = utilss.dense_to_one_hot(ori_data_label, num_class)
    label_no, label_dim = data_label_one_hot.shape
    print("label_no: ", label_no, "label_dim: ", label_dim)

    # analog encoding
    '''data_label_analog = []
    for i in ori_data_label:
        i = utilss.int2bits(i, num_class)
        i = i.numpy()
        data_label_analog.append(i[0].tolist())
    data_label_analog = np.array(data_label_analog)
    data_label_analog.reshape((-1, 1)).astype(int)'''

    network_architecture = dict(n_input=dim,
                                n_gen_1=dim - 2,
                                n_gen_2=dim - 2,
                                n_dis_1=dim - 2,
                                n_dis_2=dim - 2,
                                n_enc_1=dim - 2,
                                n_enc_2=dim - 2,
                                n_z=8,
                                n_dec_1=dim - 2,
                                n_dec_2=dim - 2)

    rmse_mvs_list = []
    rmse_cmp_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    auc_score_list = []

    data_m = np.loadtxt(args.data_dir + "data_m.txt", delimiter=",").astype(np.float32)
    data_h = utilss.data_h_gen(data_m, args.hint_rate, no, dim)
    miss_data_x = utilss.miss_data_gen(data_x, data_m)

    # data_c = data_label_numerical
    # data_c = data_label_analog
    data_c = data_label_one_hot

    for iter in np.arange(args.iter_num):
        dacin = DACINModel.DACIN(network_architecture, args.lr, args.batch_size, label_dim,
                                 model_path=args.model_path)
        DACINModel.train_model(dacin, miss_data_x, data_m, data_h, data_c, args.epochs)
        imp_x, g_x = DACINModel.imp_res_get(miss_data_x, data_m, data_c, args.model_path)

        rmse_mvs = utilss.rmse_loss(data_x, imp_x, data_m)
        rmse_cmp = utilss.rmse_loss(data_x, g_x, 1 - data_m)
        rmse_mvs_list.append(rmse_mvs)
        rmse_cmp_list.append(rmse_cmp)

        se_coef = DACINModel.get_se_coef(args.model_path)

        label_all_subjs = ori_data_label
        label_all_subjs = np.squeeze(label_all_subjs).astype(np.int64)

        C = clustering.thrC(se_coef, args.ro)
        y_x, CKSym_x = clustering.post_proC(C, args.k, args.d, args.alpha)
        y_x = y_x - 1 + ori_data_label.min()

        precision = clustering.precision(label_all_subjs, y_x)
        recall = clustering.recall(label_all_subjs, y_x)
        f1 = clustering.f1(label_all_subjs, y_x)
        roc_auc = clustering.auc(label_all_subjs, y_x, num_class)

        precision_score_list.append(precision)
        recall_score_list.append(recall)
        f1_score_list.append(f1)
        auc_score_list.append(roc_auc)

    print("rmse_mvs_list: ", rmse_mvs_list)
    print("rmse_cmp_list: ", rmse_cmp_list)
    print("precision_score_list: ", precision_score_list)
    print("recall_score_list: ", recall_score_list)
    print("f1_score_list: ", f1_score_list)
    print('auc_score_list', auc_score_list)


if __name__ == '__main__':

     main (parse_args())
