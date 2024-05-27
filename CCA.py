import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x2,x1):
        m_batchsize, _, height, width = x2.size()
        proj_query = self.query_conv(x2)
        # print(proj_query.shape)

        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,1)
        # print(proj_query_H.shape)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,1)
        # print(proj_query_W.shape)

        proj_key = self.key_conv(x2)
        # print(proj_key.shape)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # print(proj_key_H.shape)

        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # print(proj_key_W.shape)

        proj_value2 = self.value_conv(x2)
        proj_value1 = self.value_conv(x1)
        proj_value_H2 = proj_value2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)

        proj_value_W2 = proj_value2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value_H1 = proj_value1.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W1 = proj_value1.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize, width,height,height).permute(0,2,1,3)

        print(energy_H.shape)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        print(energy_W.shape)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        print(concate.shape)


        #
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        print(att_H.shape)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        print(att_W.shape)
        out_H2 = torch.bmm(proj_value_H2, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W2 = torch.bmm(proj_value_W2, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out_H1 = torch.bmm(proj_value_H1, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W1 = torch.bmm(proj_value_W1, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        # print(out_H.size(),out_W.size())

        x2=self.gamma * (out_H2 + out_W2) + x2
        x1=self.gamma * (out_H1 + out_W1) + x1
        return x2,x1

#
if __name__ == '__main__':
    model = CC_module(768)
    x1 = torch.randn(14,768, 8, 8)
    x2 =torch.randn(14,768, 8, 8)

    out = model(x2,x1)
    print(out[0].shape,out[1].shape)