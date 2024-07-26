import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='./configs/configs.yml')
    parser.add_argument('--gpu', type=str, default='mps')  # cuda:0 mps
    parser.add_argument('--float_type', type=str, default='float32')  # float64
    parser.add_argument('--epoch', type=int, default=123, help='epoch')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--model_path', type=str, default="./output/model.pt", help='output')
    parser.add_argument('--input_dim', type=int, default=100, help='input_dim')
    parser.add_argument('--n_hidden', type=int, default=200, help='n_hidden')
    parser.add_argument('--output_dim', type=int, default=10, help='output_dim')

    args = parser.parse_args()

    return args
