
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        """
        Initialize Rotary Position Embedding
        
        Args:
            dim: Dimension of the embedding (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be divisible by 2")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        
        # Create dimension indices tensor for half the dimension
        # Since we'll rotate half the dimensions, we only need dim/2
        div_term = torch.exp(
            torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2))
        )
        
        # Compute sin and cos tables for half dimensions
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, max_seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, max_seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor with positional information encoded
        """
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
            
        # Get sin and cos values for current sequence length
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # Duplicate the sin/cos for the full dimension
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1)  # [1, 1, seq_len, dim]
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
        
        # Expand to match input shape
        sin = sin.expand(batch_size, num_heads, -1, -1)
        cos = cos.expand(batch_size, num_heads, -1, -1)
        
        # Apply rotation using complex number multiplication:
        # (a + ib)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        return (x * cos) + (self._rotate_half(x) * sin)


#only difference between ffMoE and koMoE is that the router routes to linear layers rather than mlps
#I accidentally named this koMoE instead of voMoE, it only operates on the v proj and o proj of attention
class koMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        self.rank=torch.distributed.get_rank()
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        self.experts = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(num_experts)]) #deprecated feature
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(experts_per_rank)])

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        p,expert_id=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
        p=p.unsqueeze(-1) #B,S,K,1
        out=torch.empty((b,s,self.k,h))
        global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2))
        output=(p*output_local).sum(dim=2)
        return output

    
class ffMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        self.rank=torch.distributed.get_rank()
        
        self.experts = nn.ModuleList([FeedForward(hidden_size) for _ in range(num_experts)]) #deprecated feature
        
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([FeedForward(hidden_size) for _ in range(experts_per_rank)])

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        p,expert_id=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
        p=p.unsqueeze(-1) #B,S,K,1
        out=torch.empty((b,s,self.k,h))
        global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2))
        output=(p*output_local).sum(dim=2)
        return output


class MnistModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_blocks,vocab=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Initial processing
        self.initial_proj = nn.Linear(in_channels, hidden_dim)  
        
        
        #self.blocks = nn.ModuleList([
        #    Block(hidden_dim) for _ in range(num_blocks)
        #])
        block1, block2 = Block(hidden_dim), Block(hidden_dim)
        self.blocks = nn.ModuleList([block1 if i % 2 == 0 else block2 
                                       for i in range(num_blocks)]) #Univeral Transformer Parameter Sharing
        
        
        self.out_proj = nn.Linear(hidden_dim, vocab)  
        
        self.start_proj=nn.Linear(1,hidden_dim,bias=True)
        
     
    def forward(self, x,condition):
        condition=condition.to(torch.bfloat16)
        b,c,h,w=x.shape
           
        x=x.permute(0,2,3,1) #b,x,y,c
        
        x=x.reshape(b,h*w,c) 
        x = self.initial_proj(x)
        
        condition=condition.unsqueeze(-1).unsqueeze(-1)
        condition=self.start_proj(condition) #b,1,hidden_dim
        
        x=torch.cat((condition,x[:,:-1,:]),dim=1)
        
        for block in self.blocks:
            x = block(x)

        logits=self.out_proj(x)
        
        return logits #b,h*w,vocab


    @torch.no_grad()
    def generate(self, x=None, condition=1):
        condition = condition.to(torch.bfloat16)
        condition = condition.unsqueeze(-1).unsqueeze(-1)
        condition = self.start_proj(condition) #b,1,hidden_dim
        
        if x is None:
            x = condition #b,1,hidden_dim
        else:
            x = x.permute(0,2,3,1) #b,x,y,c
            x = x.reshape(b,h*w,c) 
            x = self.initial_proj(x)
            x = torch.cat((condition,x[:,:-1,:]),dim=1)
        
        max_len = 784
        generated = []
        
        for i in range(max_len):
            for block in self.blocks:
                x = block(x)
            logits = self.out_proj(x[:,-1:,:]) # Get last token
            
            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            
            token_embed = self.initial_proj(next_token.to(torch.bfloat16))
            x = torch.cat([x, token_embed], dim=1)
            generated.append(next_token)
        
       
        generated = torch.cat(generated, dim=1)
        return generated.reshape(-1, 28, 28)
            
            
        




class Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention =SHAttention(8,hidden_dim,num_heads=16)
        self.ff = ffMoE(8,hidden_dim) #num experts, hidden_size
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x



class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)




class SHAttention(nn.Module):
    def __init__(self,num_experts, hidden_dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim )
        self.k_proj=nn.Linear(hidden_dim,hidden_dim)
        #self.v_proj=nn.Linear(hidden_dim,hidden_dim)
        self.v_proj=koMoE(num_experts,hidden_dim)
        #self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj=koMoE(num_experts,hidden_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N,dim = x.shape
        #x=x.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3) #why did i split before projections
        q=self.q_proj(x)
        q=q.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=self.k_proj(x) #B,N,dim
        k=k.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        v=self.v_proj(x)
        v=v.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        #print(q.shape)
        q=q+self.rope(q)
        k=k+self.rope(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        x = self.proj(x)
        return x

