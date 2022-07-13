import torch
import torch.nn as nn
import torch.nn.functional as F


class NetEdgeHorizontal1(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal1, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeHorizontal2(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal2, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical1(nn.Module):
    def __init__(self):
        super(NetEdgeVertical1, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 1, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical2(nn.Module):
    def __init__(self):
        super(NetEdgeVertical2, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge

def weights_init_horizontal1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 3, 1, 2)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 0, 1] = 1
        m.weight.data = filter


def weights_init_horizontal2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 3, 1, 2)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 0, 1] = -1
        m.weight.data = filter


def weights_init_vertical1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 3, 2, 1)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 1, 0] = 1
        m.weight.data = filter


def weights_init_vertical2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 3, 2, 1)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 1, 0] = -1
        m.weight.data = filter

def compute_edge_for_input(map):
    conv_edge_horizontal1 = NetEdgeHorizontal1()
    conv_edge_horizontal2 = NetEdgeHorizontal2()
    conv_edge_vertical1 = NetEdgeVertical1()
    conv_edge_vertical2 = NetEdgeVertical2()

    horizontal_edge1 = conv_edge_horizontal1(map)
    horizontal_edge2 = conv_edge_horizontal2(map)
    horizontal_edge = torch.max(horizontal_edge1, horizontal_edge2)
    vertical_edge1 = conv_edge_vertical1(map)
    vertical_edge2 = conv_edge_vertical2(map)
    vertical_edge = torch.max(vertical_edge1, vertical_edge2)
    edge = torch.max(horizontal_edge, vertical_edge) > 0
    return edge.type('torch.cuda.FloatTensor')

