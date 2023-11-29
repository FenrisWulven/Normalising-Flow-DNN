import torch
from torch import nn
from torch.distributions import Distribution
from typing import List, Tuple, Callable
from torch.distributions import Dirichlet

from postnet.Encoder_Moons import Encoder_Moons
from postnet.Encoder_MNIST import Encoder_MNIST
from postnet.Encoder_CIFAR import Encoder_CIFAR

class NormalisingFlow(nn.Module):

    def __init__(self, latent: Distribution, flows: List[nn.Module]):
        super(NormalisingFlow, self).__init__()
        self.latent = latent
        self.flows = nn.ModuleList(flows)

    def latent_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_prob_z = self.latent.log_prob(z)
        return log_prob_z

    def latent_sample(self, num_samples: int = 1) -> torch.Tensor:
        return self.latent.sample((num_samples,))

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        '''Sample a new observation x by sampling z from
        the latent distribution and pass through g.'''
        return self.g(self.latent_sample(num_samples))

    def f(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Maps observation x to latent variable z.
        Additionally, computes the log determinant
        of the Jacobian for this transformation.
        Inverse of g.'''
        #z, sum_log_abs_det = x, torch.zeros(x.size(0)).to(device) # change to x device
        
        z, sum_log_abs_det = x, torch.zeros(x.size(0), device = x.device)
        for flow in self.flows: # for each transformation in the flow
            z, log_abs_det = flow.f(z)
            sum_log_abs_det += log_abs_det

        return z, sum_log_abs_det

    def g(self, z: torch.Tensor) -> torch.Tensor:
        '''Maps latent variable z to observation x.
        Inverse of f.'''
        with torch.no_grad():
            x = z
            for flow in reversed(self.flows):
                x = flow.g(x)

        return x

    def g_steps(self, z: torch.Tensor) -> List[torch.Tensor]:
        '''Maps latent variable z to observation x
        and stores intermediate results.'''
        xs = [z]
        for flow in reversed(self.flows):
            xs.append(flow.g(xs[-1]))

        return xs

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        '''Computes log p(x) using the change of variable formula.'''
        z, log_abs_det = self.f(x)
        log_prob_x = self.latent_log_prob(z) + log_abs_det
        return log_prob_x

    def __len__(self) -> int:
        return len(self.flows)

class AffineCouplingLayer(nn.Module):

    def __init__(self, theta: nn.Module, split: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        super(AffineCouplingLayer, self).__init__()
        self.theta = theta
        self.split = split

    def f(self, x: torch.Tensor) -> torch.Tensor:
        '''f: x -> z. The inverse of g.'''
        # split the input x into two halves
        x2, x1 = self.split(x) # flip the split #insert other permutation for 3D+ data
        ### permutation at one point
        t, s = self.theta(x1)
        # Tau coupling function: e^s + t
        z1, z2 = x1, x2 * torch.exp(s) + t
        # z1 and z2 are: torch.Size([64, 1]) which means that batch size is 64 and the dimension is 1
        log_det = s.sum(-1) # sum over the last dimension
        return torch.cat((z1, z2), dim=-1), log_det

    def g(self, z: torch.Tensor) -> torch.Tensor:
        '''g: z -> x. The inverse of f.'''
        z1, z2 = self.split(z)
        t, s = self.theta(z1)
        x1, x2 = z1, (z2 - t) * torch.exp(-s)
        return torch.cat((x2, x1), dim=-1)

class Conditioner(nn.Module):
    'The conditioner is the Neural Network that helps fit the model to the data by learning theta_i = (s_i,t_i)'

    def __init__(self, in_dim: int, out_dim: int, num_hidden: int, hidden_dim: int, num_params: int):
        super(Conditioner, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.LeakyReLU(inplace=True)  
        )
        self.hidden = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  
                nn.Dropout(0.3),
                nn.LeakyReLU(inplace=True)  
            )
            for _ in range(num_hidden)
        ])

        self.num_params = num_params
        self.out_dim = out_dim
        self.output = nn.Linear(hidden_dim, out_dim * num_params)
        # initialisere output lag conditioner alle vægte og bias til 0
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for h in self.hidden:
            x = h(x)

        batch_params = self.output(x).reshape(x.size(0), self.out_dim, -1)
        #batch_params[:,:,1] *= 0.001
        #batch_params[:,:,0] *= 0.001 
        params = batch_params.chunk(self.num_params, dim=-1)
        return [p.squeeze(-1) for p in params]

def get_encoder(dataset_name, latent_dim):
    if dataset_name == 'MNIST':
        return Encoder_MNIST(latent_dim)
    elif dataset_name == 'Moons':
        return Encoder_Moons(latent_dim)
    elif dataset_name == 'CIFAR':
        return Encoder_CIFAR(latent_dim)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

class PosteriorNetwork(nn.Module):
    def __init__(self, latent_dim: int, flow_models: List[nn.Module], N: torch.tensor, num_classes: int, y: torch.Tensor, reg: float, dataset_name: str):
        super(PosteriorNetwork, self).__init__()
        self.cnn = get_encoder(dataset_name, latent_dim)
        self.flow_models = nn.ModuleList(flow_models)
        self.N = N
        self.y = y
        self.num_classes = num_classes 
        self.reg = reg
    
    def forward(self, x, N):
        batch_size = x.size(0)
        # N is number of inputs in each class total
        z = self.cnn(x) 
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("NaN or Inf in tensor z")
            print(torch.where(torch.isnan(z)))
            print(z[torch.where(torch.isnan(z))])
            # Change NaN value to 0
            z[torch.where(torch.isnan(z))] = 0

            print(torch.where(torch.isinf(z)))
            print(z[torch.where(torch.isinf(z))])
            # Change Inf value to 0
            z[torch.where(torch.isinf(z))] = 0
    
        # for each class, since outputdim = num_classes
        alpha = torch.zeros((batch_size, self.num_classes)).to(z.device.type)
        log_q = torch.zeros((batch_size, self.num_classes)).to(z.device.type)
        beta_i = torch.zeros((batch_size, self.num_classes)).to(z.device.type)

        # for each class, compute 
        for c in range(self.num_classes):
            log_prob = self.flow_models[c].log_prob(z) #P(z|c,phi)
            log_q[:,c] = log_prob
            beta_prior = 1 
            #formula (4) from paper
            beta_i[:,c] = N[c] * torch.exp(log_prob) #or log_q[:,c]
            alpha[:,c] = beta_prior + beta_i[:,c] #or just beta_i[c]?

        # grad_loss
        #loss = self.loss_postnet(alpha, y, batch_size)

        ##alpha = F.normalize(alpha, p=1, dim=1) # to get p^bar_c which is the average of alphas
        #preds = self.predict(alphas) #categorical prediction using argmax on p_bar_c
        return alpha

    def loss_postnet(self, alpha, y): 
        #alpha is the p vector with alphas for each class of size: [batch x num_classes]
        #y is the ground-truth class labels of size: [batch x 1]
        batch_size = alpha.size(0)
        alpha_0 = torch.sum(alpha, dim=1, keepdim=False) #[batch]
        digamma_alpha_0 = torch.digamma(alpha_0) # batch x 1, hver obs får sin egent logprobs
        digamma_alpha_c = torch.digamma(alpha[range(batch_size),y]) # [batch] 
        # Key change: this gets the alpha value for the correct class for each obs in batch, instead of all alpha values for each obs as in the original paper
        # since we are only interested in optimising this the correct class value, and the other values are irrelevant (also because our y-labels only have one class value)
        uce_loss_elements = digamma_alpha_0 - digamma_alpha_c #changed to 0-c instead of c-0 to get positive values 
        entropy_reg = Dirichlet(alpha).entropy() #tensor of batch shape
        # Paper uses sum loss over batch - I think mean would be better since we have different batch sizes for different datasets
        postnet_loss = torch.sum(uce_loss_elements) - self.reg * torch.sum(entropy_reg) #negative since we want to minimize the loss
        return postnet_loss