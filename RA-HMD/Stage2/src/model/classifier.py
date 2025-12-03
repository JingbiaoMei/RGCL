import torch.nn as nn
import torch
### Use thi file for linear probe classifier for LCIP
class classifier_1L(nn.Module):
    def __init__(self, input_shape) -> None:
        super(classifier_1L, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1)

    def forward(self,x):
        x = self.fc1(x)
        return x

class classifier_2L(nn.Module):
    def __init__(self, input_shape, hidden_size=512) -> None:
        super(classifier_2L, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        x = self.fc2(torch.relu(self.fc1(x)))
        return x

class classifier_2L_d(nn.Module):
    def __init__(self, input_shape, hidden_size=512, input_dropout=0., dropout=0.1) -> None:
        super(classifier_2L_d, self).__init__()
        self.input_dropout = torch.nn.Dropout1d(input_dropout)
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.dropout1 = torch.nn.Dropout1d(dropout)
        self.fc2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        x = self.fc2(self.dropout1(torch.relu(self.fc1(self.input_dropout(x)))))
        return x

class classifier_3L(nn.Module):
    def __init__(self, input_shape, hidden_size_1=768, hidden_size_2=384, input_dropout=0., dropout=0.1) -> None:
        super(classifier_3L, self).__init__()
        self.input_dropout = torch.nn.Dropout1d(input_dropout)
        self.fc1 = nn.Linear(input_shape, hidden_size_1)
        self.dropout1 = torch.nn.Dropout1d(dropout)
        self.fc2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.dropout2 = torch.nn.Dropout1d(dropout)
        self.fc3 = nn.Linear(hidden_size_2,1)
    def forward(self,x):
        x = torch.relu((self.fc1(self.input_dropout(x))))
        x = self.fc3(self.dropout2(torch.relu(self.fc2(self.dropout1(x)))))
        return x

class Classifier(nn.Module):
    def __init__(self, input_shape, num_layers, proj_dim,output_dim=1, input_dropout=0., dropout=0.0) -> None:
        super(Classifier, self).__init__()
        layers = []

        
        
        # Handle dropout initialization
        if dropout is None:
            dropout = [0.0] * num_layers

        # Input dropout layer
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_shape, proj_dim))
            #nn.init.xavier_uniform_(layers[-1].weight)  # Xavier initialization
            layers.append(nn.ReLU())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_shape = proj_dim
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(proj_dim, output_dim)
    def forward(self,x, return_embed=False):
        embed = self.mlp(x)
        output = self.output_layer(embed)
        if return_embed:
            return output, embed
        return output

class Dummpy_Classifier(nn.Module):
    def __init__(self) -> None:
        super(Dummpy_Classifier, self).__init__()

        self.mlp = nn.Identity()
    def forward(self,x, return_embed=False):
        embed = self.mlp(x)
        output = torch.zeros(x.shape[0],1)
        if return_embed:
            return output, embed
        return output 
    
class classifier_hateClipper(nn.Module):
    def __init__(self, image_dim, text_dim, num_layers, proj_dim, map_dim, fusion_mode, dropout=None, batch_norm=False, args=None) -> None:
        super(classifier_hateClipper, self).__init__()
        self.fusion_mode = fusion_mode
        
        # Projection layers prior to modality fusion
        self.img_proj = nn.Sequential(nn.Linear(image_dim, map_dim), nn.Dropout(dropout[0]))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, map_dim), nn.Dropout(dropout[0]))
        #self.relu_after_fusion = nn.ReLU()
        # Modality fusion
        if fusion_mode == 'concat':
            input_shape = map_dim * 2
        elif fusion_mode == 'align':
            input_shape = map_dim
        elif fusion_mode == 'cross':
            input_shape = map_dim ** 2
        
        layers = list()
        # Append the relu after the modality fusion
        # layers.append(nn.ReLU())
        # Dropout after the modality fusion
        layers.append(nn.Dropout(dropout[1]))
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_shape, proj_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(proj_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[2]))
            
            input_shape = proj_dim
                
        self.mlp = nn.Sequential(*layers)
        if args.dataset == 'FB' or args.dataset == 'HarMeme' or args.dataset == 'MMHS' or args.dataset == 'Tamil' or args.dataset == 'HarmP':
            self.output_layer = nn.Linear(proj_dim, 1)
        elif args.dataset == 'Propaganda':
            self.output_layer = nn.Linear(proj_dim, 22)
        else:
            print("Unknown dataset: {} using binary classification by default")
            self.output_layer = nn.Linear(proj_dim, 1)
    def forward(self,img_feats, text_feats, return_embed=False):
        img_feats = self.img_proj(img_feats)
        text_feats = self.text_proj(text_feats)
        
        img_feats = nn.functional.normalize(img_feats, p=2, dim=1)
        text_feats = nn.functional.normalize(text_feats, p=2, dim=1)
        
        if self.fusion_mode == 'concat':
            x = torch.cat((img_feats, text_feats), dim=1)
        elif self.fusion_mode == 'align':
            x = torch.mul(img_feats, text_feats)
        elif self.fusion_mode == 'cross':
            x = torch.bmm(img_feats.unsqueeze(2), text_feats.unsqueeze(1)).flatten(1,2)
        
        # For embedding, we don't need the relu and dropout
        embed = self.mlp[:-2](x)
        output = self.output_layer(self.mlp(x))
        if return_embed:
            return output, embed
        return output
    