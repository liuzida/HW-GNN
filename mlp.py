import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_feats=10, h_feats=10, num_classes=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.LeakyReLU()

    def forward(self, in_feat):
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h1 = self.act(h)
        h = self.linear3(h1)
        return h, h1

class MLP_Twibot(nn.Module):
    def __init__(self, hidden_dim,
                 des_size=768,
                 tweet_size=768,
                 num_prop_size=5,
                 cat_prop_size=3,
                 dropout=0.3,
                 num_relations=2):

        super(MLP_Twibot, self).__init__()
        assert hidden_dim % 4 == 0, f"hidden_dim must be divisible by 4, got {hidden_dim}"
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim// 4, 2)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):



        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.linear_relu_output1(x)
        x = self.dropout(x)
        x1 = self.linear_relu_output2(x)
        x = self.linear_output2(x1)

        return x,x1