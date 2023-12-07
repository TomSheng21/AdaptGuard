
##################  PGD  ##################
import torch
import torchvision

class LinfStep:
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
    
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)

    # def to_image(self, x):
    #     '''
    #     Given an input (which may be in an alternative parameterization),
    #     convert it to a valid image (this is implemented as the identity
    #     function by default as most of the time we use the pixel
    #     parameterization, but for alternative parameterizations this functino
    #     must be overriden).
    #     '''
    #     return x

def replace_best(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss < loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]# .clone().detach()

    return bloss, bx

def normalize(x):
    x = torch.clamp(x, 0, 1)
    x_normalized = torchvision.transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return x_normalized

def Ent_Loss(x, target_no_use):
    softmax_x = torch.nn.Softmax(dim=1)(x)
    bs = softmax_x.size(0)
    epsilon = 1e-5
    entropy = -softmax_x * torch.log(softmax_x + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def adversarial_sample_generate(netF, netB, netC, x, iterations, eps, step_size, target=None, constraint='inf', adv_loss='ce', random_start=False, use_best=True): # eps # step_size # iterations
    '''
        Note: x must be before Normalize()
    '''
    assert iterations>0 
    assert eps>0 
    assert step_size>0

    prev_F_training = bool(netF.training)
    netF.eval()
    prev_B_training = bool(netB.training)
    netB.eval()
    prev_C_training = bool(netC.training)
    netC.eval()

    orig_input = x.detach().cuda()
    # orig_feature = netB(netF(normalize(orig_input)))
    # orig_output = netC(orig_feature)
    orig_output = netC(netB(netF(normalize(orig_input))))
    orig_pl = torch.max(orig_output, dim=1)[1]
    # netC_anchor = netC.fc.state_dict()['weight_v'][orig_pl].detach()


    # step = LinfStep(eps=0.05, orig_input=orig_input, step_size=0.01)
    if constraint == 'inf':
        step = LinfStep(eps=eps, orig_input=orig_input, step_size=step_size)
    elif constraint == '2':
        raise ValueError('Not Implement')
    else:
        raise ValueError('Unknown constraint')

    def loss_ce(target_x=None, output_adv=None, output_x=None, feature_adv=None, feature_x=None, feature_netC=None):
        assert output_adv is not None
        assert target_x is not None
        return torch.nn.CrossEntropyLoss(reduction='none')(output_adv, target_x)

    def loss_ent(target_x=None, output_adv=None, output_x=None, feature_adv=None, feature_x=None, feature_netC=None):
        assert output_adv is not None
        softmax_adv = torch.nn.Softmax(dim=1)(output_adv)
        entropy = -softmax_adv * torch.log(softmax_adv + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def loss_KLDiv(target_x=None, output_adv=None, output_x=None, feature_adv=None, feature_x=None, feature_netC=None):
        assert output_adv is not None
        assert output_x is not None
        KLDiv = nn.KLDivLoss(reduction='none')(F.log_softmax(output_adv, dim=1), F.softmax(output_x, dim=1))
        return torch.sum(KLDiv, dim=1) * output_x.shape[0]

    def cosine_distance_loss(a,b):
        # a,b (Batch_size * Dim) --> (Batch_size)
        return 1-torch.sum((a/torch.norm(a,dim=1,keepdim=True))*(b/torch.norm(b,dim=1,keepdim=True)),dim=1)

    def loss_cosine(target_x=None, output_adv=None, output_x=None, feature_adv=None, feature_x=None, feature_netC=None):
        assert feature_adv is not None
        assert feature_x is not None
        return cosine_distance_loss(feature_adv, feature_x)

    def loss_cosine_netC(target_x=None, output_adv=None, output_x=None, feature_adv=None, feature_x=None, feature_netC=None):
        assert feature_adv is not None
        assert feature_netC is not None
        return cosine_distance_loss(feature_adv, feature_netC)


   

    if adv_loss == 'ce':
        loss_function = loss_ce
    elif adv_loss == 'ent':
        loss_function = loss_ent
    elif adv_loss == 'KLDiv':
        loss_function = loss_KLDiv
    # elif adv_loss == 'cosine':
    #     loss_function = loss_cosine
    # elif adv_loss == 'cosine_prototype':
    #     loss_function = loss_cosine_netC
    else:
        raise ValueError('Unknown loss')
    
    best_loss = None
    best_x_adv = None

    if random_start:
        x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()


    for _ in range(iterations):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        # print(x_adv[0])
        # feature_adv = netB(netF(normalize(x_adv)))
        # output_adv = netC(feature_adv)
        output_adv = netC(netB(netF(normalize(x_adv))))
        losses = loss_function(target_x=target, output_adv=output_adv, output_x=orig_output, feature_adv=None, feature_x=None, feature_netC=None)
        
        assert losses.shape[0] == x_adv.shape[0], 'Shape of losses must match input!'
        loss = torch.mean(losses)
        # print(loss)
        grad = torch.autograd.grad(loss, [x_adv])[0]

        # print(grad)
        # import sys
        # sys.exit()

        with torch.no_grad():
            best_loss, best_x_adv = replace_best(losses, best_loss, x_adv, best_x_adv) #  use_best else (losses, x)
            x_adv = step.step(x_adv, grad)
            x_adv = step.project(x_adv)

    # feature_adv = netB(netF(normalize(x_adv)))
    # output_adv = netC(feature_adv)
    output_adv = netC(netB(netF(normalize(x_adv))))
    losses = loss_function(target_x=target, output_adv=output_adv, output_x=orig_output, feature_adv=None, feature_x=None, feature_netC=None)
    with torch.no_grad():
        best_loss, best_x_adv = replace_best(losses, best_loss, x_adv, best_x_adv)

    if prev_F_training:
        netF.train()
        # print('netF back to train')
    if prev_B_training:
        netB.train()
        # print('netB back to train')
    if prev_C_training:
        netC.train()
        # print('netC back to train')

    if use_best:
        return best_x_adv #  if return_image else best_x
    else:
        return x_adv



