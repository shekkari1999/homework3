import torch
import torch.nn as nn
from config import hidden_size, num_layers, dropout, emb_dim, vocab_size


class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size = vocab_size, emb_dim = emb_dim, hidden_size = hidden_size, 
                    num_layers = num_layers, dropout = dropout, activation='tanh'):
        '''
        Initialize Wxh's for all layers. For input layer this will be (emb_dim, hidden_size), but for
        all other's this will be (hidden_size, hidden_size)
        '''
        super().__init__()
        self.dropout_prob = dropout 
        self.activation = activation 
        self.Wxhs = nn.ModuleList()
        self.Whhs = nn.ModuleList()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        # Define dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        for l in range(num_layers):
            '''
            for each layer, if input is coming straight from data, do infeatures = emd_dim, else its
            all hidden units
            '''
            if l == 0:
                self.Wxhs.append(nn.Linear(in_features = emb_dim, out_features = hidden_size))
                self.Whhs.append(nn.Linear(in_features = hidden_size, out_features = hidden_size))
            else:
                self.Wxhs.append(nn.Linear(in_features = hidden_size, out_features = hidden_size))
                self.Whhs.append(nn.Linear(in_features = hidden_size, out_features = hidden_size))
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # Pass input through the embedding layer
        x = self.embeddings(x) # x becomes (batch_size, sequence_length, emb_dim)

        batch, seq_len, _ = x.size()
        '''
        one h list for each layer. this is not Whh, these are computed hidden states. That's why this is in forward
        '''
        # since we are creating a new tensor here, always be mindful to include device
        h = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)]
        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                if l == 0:
                    inp = x_t
                else:
                    inp = h[l - 1]

                # Apply the specified activation function
                if self.activation == 'tanh':
                    h[l] = torch.tanh(self.Wxhs[l](inp) + self.Whhs[l](h[l]))
                elif self.activation == 'relu':
                     h[l] = torch.relu(self.Wxhs[l](inp) + self.Whhs[l](h[l]))
                elif self.activation == 'sigmoid':
                     h[l] = torch.sigmoid(self.Wxhs[l](inp) + self.Whhs[l](h[l]))
                else:
                    # Default to tanh if activation is not recognized
                    h[l] = torch.tanh(self.Wxhs[l](inp) + self.Whhs[l](h[l]))
                # Apply dropout to the output of each layer except the last
                if l < self.num_layers - 1:
                    h[l] = self.dropout(h[l])

        out = torch.sigmoid(self.fc(h[-1]))

        return out

class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size = vocab_size, emb_dim = emb_dim, hidden_size = hidden_size, 
                    num_layers = num_layers, dropout = dropout, activation='tanh' ):
        super().__init__()
        '''
        All the layers will have a forget matrix, input matrix, candidate matrix and output matrix 
        for both hidden state and input state
        '''
        self.Wfhs = nn.ModuleList() ## for forget(hidden)
        self.Wfxs = nn.ModuleList() ## for forget(input)
        self.Wihs = nn.ModuleList() ## for input (hidden)
        self.Wixs = nn.ModuleList() ## for input (input)
        self.Wghs = nn.ModuleList() ## for candidate (hidden)
        self.Wgxs = nn.ModuleList() ## for candidate (input)
        self.Wohs = nn.ModuleList() ## for output (hidden)
        self.Woxs = nn.ModuleList() ## for output (input)

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.activation = activation 
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)
        # Now, we gotta initialize those modules according to corresponding layers

        for l in range(self.num_layers):
            if l == 0:
                self.Wfxs.append(nn.Linear(emb_dim, hidden_size))
                self.Wixs.append(nn.Linear(emb_dim, hidden_size))
                self.Wgxs.append(nn.Linear(emb_dim, hidden_size))
                self.Woxs.append(nn.Linear(emb_dim, hidden_size))
                self.Wfhs.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.Wfxs.append(nn.Linear(hidden_size, hidden_size))
                self.Wixs.append(nn.Linear(hidden_size, hidden_size))
                self.Wgxs.append(nn.Linear(hidden_size, hidden_size))
                self.Woxs.append(nn.Linear(hidden_size, hidden_size))
                self.Wfhs.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x = self.embedding_layer(x)
        batch, seq_len, _ = x.size()

        ### for all layers of input at timestamp t, initialize h, c with zeros
        h = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)] 
        c = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)]

        # for each timestamp, pick the word
        for t in range(seq_len):
            x_t = x[:, t, :] 
            
            ## this will be input to only ground layer
            for l in range(self.num_layers):
                if l == 0:
                    inp = x_t 
                else:
                    inp = h[l - 1]
                ## calculate the four gates
                forget_gate = torch.sigmoid(self.Wfxs[l](inp) + self.Wfhs[l](h[l]))
                input_gate  = torch.sigmoid(self.Wixs[l](inp) + self.Wihs[l](h[l]))
                if self.activation == 'tanh':
                    candidate = torch.tanh(self.Wgxs[l](inp) + self.Wghs[l](h[l]))
                elif self.activation == 'relu':
                    candidate = torch.relu(self.Wgxs[l](inp) + self.Wghs[l](h[l]))
                elif self.activation == 'sigmoid':
                    candidate = torch.sigmoid(self.Wgxs[l](inp) + self.Wghs[l](h[l]))
                output_gate = torch.sigmoid(self.Woxs[l](inp) + self.Wohs[l](h[l]))
                ## compute cell state
                c[l] = forget_gate * c[l] + input_gate * candidate 
                if self.activation == 'tanh':
                    h[l] = output_gate * torch.tanh(c[l])
                elif self.activation == 'relu':
                    h[l] = output_gate * torch.relu(c[l])
                elif self.activation == 'sigmoid':
                    h[l] = output_gate * torch.sigmoid(c[l])
                  # Apply dropout to the output of each layer except the last
                if l < self.num_layers - 1:
                    h[l] = self.dropout(h[l])

        out = torch.sigmoid(self.fc(h[-1]))

        return out


class MultiLayerBiDrectionalLSTM(nn.Module):
    def __init__(self, vocab_size = vocab_size, emb_dim = emb_dim, hidden_size = hidden_size, 
                    num_layers = num_layers, dropout = dropout, activation='tanh'):
        super().__init__()
        '''
        All the layers will have a foward and backward forget matrix, input matrix, candidate matrix and output matrix 
        for both hidden state and input state
        '''
        self.Wfhs_f = nn.ModuleList() ## for forget(hidden)
        self.Wfxs_f = nn.ModuleList() ## for forget(input)
        self.Wihs_f = nn.ModuleList() ## for input (hidden)
        self.Wixs_f = nn.ModuleList() ## for input (input)
        self.Wghs_f = nn.ModuleList() ## for candidate (hidden)
        self.Wgxs_f = nn.ModuleList() ## for candidate (input)
        self.Wohs_f = nn.ModuleList() ## for output (hidden)
        self.Woxs_f = nn.ModuleList() ## for output (input)

        # backward
        self.Wfhs_b = nn.ModuleList() ## for forget(hidden)
        self.Wfxs_b = nn.ModuleList() ## for forget(input)
        self.Wihs_b = nn.ModuleList() ## for input (hidden)
        self.Wixs_b = nn.ModuleList() ## for input (input)
        self.Wghs_b = nn.ModuleList() ## for candidate (hidden)
        self.Wgxs_b = nn.ModuleList() ## for candidate (input)
        self.Wohs_b = nn.ModuleList() ## for output (hidden)
        self.Woxs_b = nn.ModuleList() ## for output (input)

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.activation = activation 
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(2 * hidden_size, 1)


        for l in range(self.num_layers):
            if l == 0:
                self.Wfxs_f.append(nn.Linear(emb_dim, hidden_size))
                self.Wixs_f.append(nn.Linear(emb_dim, hidden_size))
                self.Wgxs_f.append(nn.Linear(emb_dim, hidden_size))
                self.Woxs_f.append(nn.Linear(emb_dim, hidden_size))
                self.Wfhs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wfxs_b.append(nn.Linear(emb_dim, hidden_size))
                self.Wixs_b.append(nn.Linear(emb_dim, hidden_size))
                self.Wgxs_b.append(nn.Linear(emb_dim, hidden_size))
                self.Woxs_b.append(nn.Linear(emb_dim, hidden_size))
                self.Wfhs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs_b.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.Wfxs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wixs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wgxs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Woxs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wfhs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs_f.append(nn.Linear(hidden_size, hidden_size))
                self.Wfxs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wixs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wgxs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Woxs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wfhs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wihs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wghs_b.append(nn.Linear(hidden_size, hidden_size))
                self.Wohs_b.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x = self.embedding_layer(x)
        batch, seq_len, _ = x.size()

        ### for all layers of input at timestamp t, initialize h, c for forward and backward with zeros
        h_f = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)] 
        c_f = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)]
        h_b = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)] 
        c_b = [torch.zeros(batch, self.hidden_size, device = x.device) for _ in range(self.num_layers)]

        '''
        In normal lstm, the last hidden state summarizes everything well. but in bilstm, 
        we do need outputs at each timestep to align later.
        '''
        outputs_f, outputs_b = [], []

        # for each timestamp, pick the word and also reverse and pick the word
        for t in range(seq_len):

            x_t   =   x[:, t, :] 
            x_t_b =   x[:, seq_len - t - 1, :]

            ## this will be input to only ground layer
            for l in range(self.num_layers):
                if l == 0:
                    inp_f = x_t 
                    inp_b = x_t_b
                else:
                    inp_f = h_f[l - 1]
                    inp_b = h_b[l - 1]

             ## calculate the four gates forward and backward
                forget_gate_f = torch.sigmoid(self.Wfxs_f[l](inp_f) + self.Wfhs_f[l](h_f[l]))
                input_gate_f  = torch.sigmoid(self.Wixs_f[l](inp_f) + self.Wihs_f[l](h_f[l]))
                forget_gate_b = torch.sigmoid(self.Wfxs_b[l](inp_b) + self.Wfhs_b[l](h_b[l]))
                input_gate_b  = torch.sigmoid(self.Wixs_b[l](inp_b) + self.Wihs_b[l](h_b[l]))
                if self.activation == 'tanh':
                    candidate_f = torch.tanh(self.Wgxs_f[l](inp_f) + self.Wghs_f[l](h_f[l]))
                    candidate_b = torch.tanh(self.Wgxs_b[l](inp_b) + self.Wghs_b[l](h_b[l]))

                elif self.activation == 'relu':
                    candidate_f = torch.relu(self.Wgxs_f[l](inp_f) + self.Wghs_f[l](h_f[l]))
                    candidate_b = torch.relu(self.Wgxs_b[l](inp_b) + self.Wghs_b[l](h_b[l]))
                elif self.activation == 'sigmoid':
                    candidate_f = torch.sigmoid(self.Wgxs_f[l](inp_f) + self.Wghs_f[l](h_f[l]))
                    candidate_b = torch.sigmoid(self.Wgxs_b[l](inp_b) + self.Wghs_b[l](h_b[l]))

                output_gate_f = torch.sigmoid(self.Woxs_f[l](inp_f) + self.Wohs_f[l](h_f[l]))
                output_gate_b = torch.sigmoid(self.Woxs_b[l](inp_b) + self.Wohs_b[l](h_b[l]))

                ## compute cell state for forward and back 
                c_f[l] = forget_gate_f * c_f[l] + input_gate_f * candidate_f 
                c_b[l] = forget_gate_b * c_b[l] + input_gate_b * candidate_b 

                if self.activation == 'tanh':
                    h_f[l] = output_gate_f * torch.tanh(c_f[l])
                    h_b[l] = output_gate_b * torch.tanh(c_b[l])

                elif self.activation == 'relu':
                    h_f[l] = output_gate_f * torch.relu(c_f[l])
                    h_b[l] = output_gate_b * torch.relu(c_b[l])
                elif self.activation == 'sigmoid':
                    h_f[l] = output_gate_f * torch.sigmoid(c_f[l])
                    h_b[l] = output_gate_b * torch.sigmoid(c_b[l])
                  # Apply dropout to the output of each layer except the last
                if l < self.num_layers - 1:
                    h_f[l] = self.dropout(h_f[l])
                    h_b[l] = self.dropout(h_b[l])
            outputs_f.append(h_f[-1])
            outputs_b.append(h_b[-1])

        # Merge forward/backward sequences 
        h_f_seq = torch.stack(outputs_f, dim=1)
        h_b_seq = torch.stack(outputs_b[::-1], dim=1)  # reverse to align
        h_final = torch.cat([h_f_seq[:, -1, :], h_b_seq[:, 0, :]], dim=-1)

        out = torch.sigmoid(self.fc(h_final))
        return out





