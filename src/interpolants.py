import numpy as np
import torch
import json
from scipy.interpolate import make_interp_spline



###### Linear Interpolant ######

# def linear_interpolant(values, times, mask):
#     filled = values.clone()
#     for j in range(values.shape[1]):
#         obs_idx = np.where(mask[:, j])[0]
#         if obs_idx.size == 0:
#             filled[:, j] = 0.0
#             continue
#         filled[:, j] = np.interp(
#             times,
#             times[obs_idx],
#             values[obs_idx, j],
#             left=values[obs_idx[0], j],
#             right=values[obs_idx[-1], j],
#         )
#     return filled


# def linear_interpolant(values, times, mask):
#     filled = values.clone()
#     batch, steps, dims = values.shape
#     for b in range(batch):
#         for d in range(dims):
#             obs = mask[b, :, d].bool()
#             if obs.sum() == 0:
#                 filled[b, :, d] = 0.0
#                 continue
#             t_obs = times[b, obs]
#             v_obs = values[b, obs, d]
#             for i in range(steps):
#                 if obs[i]:
#                     continue
#                 t_i = times[b, i]
#                 idx = torch.searchsorted(t_obs, t_i)
#                 if idx == 0:
#                     filled[b, i, d] = v_obs[0]
#                 elif idx >= t_obs.shape[0]:
#                     filled[b, i, d] = v_obs[-1]
#                 else:
#                     t_left, t_right = t_obs[idx - 1], t_obs[idx]
#                     v_left, v_right = v_obs[idx - 1], v_obs[idx]
#                     w = (t_i - t_left) / (t_right - t_left)
#                     filled[b, i, d] = v_left + w * (v_right - v_left)
#     return filled

# def linear_interpolant_velocity_field(values, times, mask, subsample_per_interval=32):

#     batch_size, time_steps, data_dim = values.shape
#     xt = []
#     vt = []
#     t = []

#     for step in range(time_steps - 1):
#         l_t = values[:, step, :]
#         r_t = values[:, step + 1, :]
#         time_interval = (times[:, step + 1] - times[:, step]).unsqueeze(-1)
#         slope = (r_t - l_t) / time_interval
#         for i in range(subsample_per_interval):
#             t_i = times[:, step].unsqueeze(-1) + i * time_interval / subsample_per_interval
#             x_i = l_t + i * slope * time_interval / subsample_per_interval
#             v_i = slope
#             t.append(t_i)
#             xt.append(x_i)
#             vt.append(v_i)

#     return torch.stack(xt, dim=1), torch.stack(vt, dim=1), torch.stack(t, dim=1)


###### B-spline Interpolant ######
# def bspline_interpolant(values, times, mask, degree=3):
#     filled = values.clone()
#     batch, steps, dims = values.shape
#     for b in range(batch):
#         t_all = times[b].detach().cpu().numpy()
#         for d in range(dims):
#             obs = mask[b, :, d].bool()
#             if obs.sum() == 0:
#                 filled[b, :, d] = 0.0
#                 continue
#             t_obs = t_all[obs.cpu().numpy()]
#             v_obs = values[b, obs, d].detach().cpu().numpy()
#             k = min(degree, t_obs.size - 1)
#             if k <= 0:
#                 filled[b, obs, d] = values[b, obs, d]
#                 filled[b, ~obs, d] = values[b, obs, d][0]
#                 continue
#             spline = make_interp_spline(t_obs, v_obs, k=k, axis=0)
#             filled_vals = spline(t_all)
#             filled[b, :, d] = torch.as_tensor(filled_vals, dtype=values.dtype, device=values.device)
#     return filled


# def bspline_interpolant_velocity_field(values, times, mask, subsample_per_interval=32, degree=3):
#     # assume `values` already came from `bspline_interpolant`
#     batch_size, time_steps, data_dim = values.shape
#     xt, vt, tt = [], [], []
#     for b in range(batch_size):
#         t_all = times[b].detach().cpu().numpy()
#         for d in range(data_dim):
#             v_all = values[b, :, d].detach().cpu().numpy()
#             spline = make_interp_spline(t_all, v_all, k=min(degree, time_steps - 1), axis=0)
#             spline_d = spline.derivative()
#             for step in range(time_steps - 1):
#                 t_left = times[b, step]
#                 t_right = times[b, step + 1]
#                 delta = (t_right - t_left) / subsample_per_interval
#                 for i in range(subsample_per_interval):
#                     t_i = t_left + i * delta
#                     if d == 0: # only append time once
#                         tt.append(t_i.unsqueeze(0))

#                     x_at_ti = spline(t_i.cpu().numpy())
#                     xt.append(torch.as_tensor(x_at_ti, dtype=values.dtype, device=values.device).unsqueeze(0))

#                     vx = spline_d(t_i.cpu().numpy())
#                     vt.append(torch.as_tensor(vx, dtype=values.dtype, device=values.device).unsqueeze(0))

#     xt = torch.cat(xt, dim=0).view(batch_size, -1, data_dim)
#     vt = torch.cat(vt, dim=0).view(batch_size, -1, data_dim)
#     tt = torch.stack(tt, dim=0).view(batch_size, -1, 1)
#     return xt, vt, tt


# ###### Cubic Interpolant ######
# def cubic_interpolant(values, times, mask):
#     pass

# def cubic_interpolant_velocity_field(values, times, mask, subsample_per_interval=32):
#     pass


# ###### Lagrange Interpolant ######
# def lagrange_interpolant(values, times, mask):
#     pass

# def lagrange_interpolant_velocity_field(values, times, mask, subsample_per_interval=32):
#     pass



# def bspline_interpolant_values(values, times, mask, degree=3, subsample_per_interval=2, dynamics_kind='ode', sigma=0.001):
#     # filled = values.clone()
#     batch, steps, dims = values.shape
#     total_refined_steps= steps*subsample_per_interval

#     xt= torch.zeros((batch, total_refined_steps, dims))
#     vt= torch.zeros((batch, total_refined_steps, dims))
#     tt= torch.zeros((batch, total_refined_steps, 1))
#     eps_xt= torch.zeros((batch, total_refined_steps, dims))
#     eps_score= torch.zeros((batch, total_refined_steps, dims))
#     lambda_t= torch.zeros((batch, total_refined_steps, dims))
#     sigma_t= torch.zeros((batch, total_refined_steps, dims))
#     der_sigma_t= torch.zeros((batch, total_refined_steps, dims))
#     score= torch.zeros((batch, total_refined_steps, dims))
#     for b in range(batch):
#         value_traj= values[b].detach().cpu()
#         times_traj= times[b].detach().cpu()
#         mask_traj= mask[b].detach().cpu()
#         start = times_traj[0].detach().cpu()
#         end   = times_traj[-1].detach().cpu()
#         if dynamics_kind=='sde_quadratic_sigma':
#             t_fine = np.random.uniform(low=start, high=end, size=total_refined_steps).astype(np.float32)
#             t_fine.sort()
#         else:
#             t_fine = np.linspace(start, end, total_refined_steps)
#         for d in range(dims):
#             dim_idx= np.where(mask_traj[:, d])
#             dim_value_traj_observed_samples= value_traj[:, d][dim_idx]
#             dim_spline = make_interp_spline(times_traj[dim_idx], dim_value_traj_observed_samples, k=degree, axis=0)

#             dim_spline_curve= dim_spline(t_fine)

#             spline_der = dim_spline.derivative()   # first derivative spline
#             dim_spline_velocities = spline_der(t_fine)  # numpy array of shape (total_refined_steps,)


#             ## old
#             # xt[b, :, d]= torch.from_numpy(dim_spline_curve) ## mu_t
#             # vt[b, :, d]= torch.from_numpy(dim_spline_velocities) ## der_mu_t

#             ## added new
#             # mu_t_d= torch.from_numpy(dim_spline_curve)
#             # der_mu_t_d= torch.from_numpy(dim_spline_velocities)
#             # vt_d, score_d, xt_d, eps_xt_d, eps_score_d, lambda_t_d= get_velocity_score_and_noise(der_mu_t_d, mu_t_d, dim_value_traj_observed_samples, times_traj[dim_idx], t_fine, dynamics_kind, interpolant_kind='bspline', sigma=sigma)
            

#             # xt[b, :, d]= xt_d
#             # vt[b, :, d]= vt_d
            
#             # eps_xt[b, :, d]= eps_xt_d
#             # eps_score[b, :, d]= eps_score_d

#             # lambda_t[b, :, d]= lambda_t_d
#             # score[b, :, d]= score_d

#             ## added newest
#             mu_t_d= torch.from_numpy(dim_spline_curve)
#             der_mu_t_d= torch.from_numpy(dim_spline_velocities)
#             sigma_t_d, lambda_t_d, der_sigma_t_d= get_velocity_score_and_noise(der_mu_t_d, mu_t_d, dim_value_traj_observed_samples, times_traj[dim_idx], t_fine, dynamics_kind, interpolant_kind='bspline', sigma=sigma)
            
#             xt[b, :, d]= mu_t_d
#             vt[b, :, d]= der_mu_t_d

#             lambda_t[b, :, d]= lambda_t_d
#             sigma_t[b, :, d]= sigma_t_d
#             der_sigma_t[b, :, d]= der_sigma_t_d



        
#         tt[b, :, 0]= torch.from_numpy(t_fine) 

#     # Move results back to the original device (values/times may live on GPU)
#     device = values.device
#     # return xt.to(device), vt.to(device), tt.to(device), score.to(device), lambda_t.to(device), eps_xt.to(device), eps_score.to(device)
#     return xt.to(device), vt.to(device), tt.to(device), sigma_t.to(device), lambda_t.to(device), der_sigma_t.to(device)
#     # return xt.to(device), vt.to(device), tt.to(device)



def bspline_interpolant_values(values, times, mask, degree=3, subsample_per_interval=2, dynamics_kind='ode', sigma=0.001):
    """
    B-spline interpolant with proper handling of degree=1 to match linear interpolation.
    
    - degree == 1: Uses np.interp (exactly matches linear_interpolant_values)
    - degree  > 1: Uses scipy B-spline with flat extrapolation outside observed range
    """
    batch, steps, dims = values.shape
    total_refined_steps = steps * subsample_per_interval

    xt = torch.zeros((batch, total_refined_steps, dims))
    vt = torch.zeros((batch, total_refined_steps, dims))
    tt = torch.zeros((batch, total_refined_steps, 1))
    lambda_t = torch.zeros((batch, total_refined_steps, dims))
    sigma_t = torch.zeros((batch, total_refined_steps, dims))
    der_sigma_t = torch.zeros((batch, total_refined_steps, dims))
    
    for b in range(batch):
        value_traj = values[b].detach().cpu()
        times_traj = times[b].detach().cpu()
        mask_traj = mask[b].detach().cpu()
        
        start = times_traj[0].detach().cpu().item()
        end = times_traj[-1].detach().cpu().item()
        
        if dynamics_kind == 'sde_quadratic_sigma':
            t_fine = np.random.uniform(low=start, high=end, size=total_refined_steps).astype(np.float32)
            t_fine.sort()
        else:
            t_fine = np.linspace(start, end, total_refined_steps)
            
        for d in range(dims):
            dim_idx = np.where(mask_traj[:, d])[0]
            
            if dim_idx.size == 0:
                # No observations for this dimension
                xt[b, :, d] = 0.0
                vt[b, :, d] = 0.0
                continue
                
            dim_value_traj_observed_samples = value_traj[:, d][dim_idx].numpy()
            t_obs = times_traj[dim_idx].numpy()
            
            # Handle single observation
            if dim_idx.size == 1:
                xt[b, :, d] = torch.from_numpy(np.full_like(t_fine, dim_value_traj_observed_samples[0]))
                vt[b, :, d] = 0.0
                sigma_t[b, :, d] = sigma if dynamics_kind != 'ode' else 0.0
                lambda_t[b, :, d] = (2.0 / (sigma**2)) * sigma_t[b, :, d] if dynamics_kind != 'ode' else 0.0
                der_sigma_t[b, :, d] = 0.0
                continue
            
            # FIX 1: For degree=1, use exact linear interpolation (matches linear_interpolant_values)
            if degree == 1:
                dim_linear_curve = np.interp(t_fine, t_obs, dim_value_traj_observed_samples)
                
                # Compute velocities using finite differences (same as linear_interpolant_values)
                dim_linear_velocities = (dim_linear_curve[1:] - dim_linear_curve[:-1]) / (t_fine[1:] - t_fine[:-1])
                last_velocity = (dim_linear_curve[-1] - dim_linear_curve[-2]) / (t_fine[-1] - t_fine[-2])
                dim_linear_velocities = np.append(dim_linear_velocities, float(last_velocity))
                
                mu_t_d = torch.from_numpy(dim_linear_curve)
                der_mu_t_d = torch.from_numpy(dim_linear_velocities)
                
                # Use 'linear' for interpolant_kind to match linear_interpolant_values exactly
                sigma_t_d, lambda_t_d, der_sigma_t_d = get_velocity_score_and_noise(
                    der_mu_t_d, mu_t_d, 
                    torch.from_numpy(dim_value_traj_observed_samples), 
                    torch.from_numpy(t_obs), 
                    t_fine, 
                    dynamics_kind, 
                    interpolant_kind='linear',  # Key: use 'linear' not 'bspline' for degree=1
                    sigma=sigma
                )
                
                xt[b, :, d] = mu_t_d
                vt[b, :, d] = der_mu_t_d
                lambda_t[b, :, d] = lambda_t_d
                sigma_t[b, :, d] = sigma_t_d
                der_sigma_t[b, :, d] = der_sigma_t_d
                continue
            
            # FIX 2: For degree > 1, use B-spline with clamped extrapolation
            k = min(degree, dim_idx.size - 1)
            if k <= 0:
                # Fallback to constant
                xt[b, :, d] = torch.from_numpy(np.full_like(t_fine, dim_value_traj_observed_samples[0]))
                vt[b, :, d] = 0.0
                sigma_t[b, :, d] = sigma if dynamics_kind != 'ode' else 0.0
                lambda_t[b, :, d] = (2.0 / (sigma**2)) * sigma_t[b, :, d] if dynamics_kind != 'ode' else 0.0
                der_sigma_t[b, :, d] = 0.0
                continue
            
            # Fit spline on observed points
            dim_spline = make_interp_spline(t_obs, dim_value_traj_observed_samples, k=k, axis=0)
            
            # Get observed time range
            t_obs_min, t_obs_max = t_obs[0], t_obs[-1]
            
            # Clamp evaluation times to observed range
            t_eval = np.clip(t_fine, t_obs_min, t_obs_max)
            
            # Evaluate spline
            dim_spline_curve = dim_spline(t_eval)
            spline_der = dim_spline.derivative()
            dim_spline_velocities = spline_der(t_eval)
            
            # Enforce flat extrapolation outside observed range (like np.interp)
            left_mask = t_fine < t_obs_min
            right_mask = t_fine > t_obs_max
            
            if np.any(left_mask):
                dim_spline_curve[left_mask] = dim_value_traj_observed_samples[0]
                dim_spline_velocities[left_mask] = 0.0
            if np.any(right_mask):
                dim_spline_curve[right_mask] = dim_value_traj_observed_samples[-1]
                dim_spline_velocities[right_mask] = 0.0
            
            mu_t_d = torch.from_numpy(dim_spline_curve)
            der_mu_t_d = torch.from_numpy(dim_spline_velocities)
            
            sigma_t_d, lambda_t_d, der_sigma_t_d = get_velocity_score_and_noise(
                der_mu_t_d, mu_t_d, 
                torch.from_numpy(dim_value_traj_observed_samples), 
                torch.from_numpy(t_obs), 
                t_fine, 
                dynamics_kind, 
                interpolant_kind='bspline', 
                sigma=sigma
            )
            
            xt[b, :, d] = mu_t_d
            vt[b, :, d] = der_mu_t_d
            lambda_t[b, :, d] = lambda_t_d
            sigma_t[b, :, d] = sigma_t_d
            der_sigma_t[b, :, d] = der_sigma_t_d
        
        tt[b, :, 0] = torch.from_numpy(t_fine)
    
    # Move results back to original device
    device = values.device
    return xt.to(device), vt.to(device), tt.to(device), sigma_t.to(device), lambda_t.to(device), der_sigma_t.to(device)







def linear_interpolant_values(values, times, mask, subsample_per_interval=2, dynamics_kind='ode', sigma=0.001):
    # filled = values.clone()

    if len(values.shape) == 5:
        batch, steps, height, width, channels = values.shape
        values= values.view(batch, steps, -1)
        mask= mask.view(batch, steps, -1)
        dims= channels*height*width
    else:
        batch, steps, dims = values.shape

    total_refined_steps= steps*subsample_per_interval

    xt= torch.zeros((batch, total_refined_steps, dims))
    vt= torch.zeros((batch, total_refined_steps, dims))
    tt= torch.zeros((batch, total_refined_steps, 1))
    eps_xt= torch.zeros((batch, total_refined_steps, dims))
    eps_score= torch.zeros((batch, total_refined_steps, dims))
    lambda_t= torch.zeros((batch, total_refined_steps, dims))
    sigma_t= torch.zeros((batch, total_refined_steps, dims))
    der_sigma_t= torch.zeros((batch, total_refined_steps, dims))
    score= torch.zeros((batch, total_refined_steps, dims))
    for b in range(batch):
        value_traj= values[b].detach().cpu()
        times_traj= times[b].detach().cpu()
        mask_traj= mask[b].detach().cpu()
        start = times_traj[0].detach().cpu().item()
        end   = times_traj[-1].detach().cpu().item()
        if dynamics_kind=='sde_quadratic_sigma':
            t_fine = np.random.uniform(low=start, high=end, size=total_refined_steps).astype(np.float32)
            t_fine.sort()
        else:
            t_fine = np.linspace(start, end, total_refined_steps)
        for d in range(dims):
            dim_idx= np.where(mask_traj[:, d])
            dim_value_traj_observed_samples= value_traj[:, d][dim_idx]
            dim_linear_curve = np.interp(t_fine, times_traj[dim_idx], dim_value_traj_observed_samples)

            dim_linear_velocities= (dim_linear_curve[1:]-dim_linear_curve[:-1])/(t_fine[1:]-t_fine[:-1])
            last_velocity= (dim_linear_curve[-1]-dim_linear_curve[-2])/(t_fine[-1]-t_fine[-2])
            
            ## old
            # xt[b, :, d]= torch.from_numpy(dim_linear_curve)
            # vt[b, :-1, d]= torch.from_numpy(dim_linear_velocities)
            # vt[b, -1, d]= float(last_velocity)

            ## added new
            # mu_t_d= torch.from_numpy(dim_linear_curve)
            # dim_linear_velocities= np.append(dim_linear_velocities, float(last_velocity))
            # der_mu_t_d= torch.from_numpy(dim_linear_velocities)
            # vt_d, score_d, xt_d, eps_xt_d, eps_score_d, lambda_t_d= get_velocity_score_and_noise(der_mu_t_d, mu_t_d, dim_value_traj_observed_samples, times_traj[dim_idx], t_fine, dynamics_kind, interpolant_kind='linear', sigma=sigma)

            # xt[b, :, d]= xt_d
            # vt[b, :, d]= vt_d
            
            # eps_xt[b, :, d]= eps_xt_d
            # eps_score[b, :, d]= eps_score_d

            # lambda_t[b, :, d]= lambda_t_d
            # score[b, :, d]= score_d


            ## added newest
            mu_t_d= torch.from_numpy(dim_linear_curve)
            dim_linear_velocities= np.append(dim_linear_velocities, float(last_velocity))
            der_mu_t_d= torch.from_numpy(dim_linear_velocities)
            sigma_t_d, lambda_t_d, der_sigma_t_d= get_velocity_score_and_noise(der_mu_t_d, mu_t_d, dim_value_traj_observed_samples, times_traj[dim_idx], t_fine, dynamics_kind, interpolant_kind='linear', sigma=sigma)
            
            xt[b, :, d]= mu_t_d
            vt[b, :, d]= der_mu_t_d

            lambda_t[b, :, d]= lambda_t_d
            sigma_t[b, :, d]= sigma_t_d
            der_sigma_t[b, :, d]= der_sigma_t_d

        tt[b, :, 0]= torch.from_numpy(t_fine) 

    device = values.device

    if len(values.shape) == 5:
        xt= xt.view(batch, steps, height, width, channels)
        vt= vt.view(batch, steps, height, width, channels)
        sigma_t= sigma_t.view(batch, steps, height, width, channels)
        lambda_t= lambda_t.view(batch, steps, height, width, channels)
    # return xt.to(device), vt.to(device), tt.to(device)
    return xt.to(device), vt.to(device), tt.to(device), sigma_t.to(device), lambda_t.to(device), der_sigma_t.to(device)
    # return xt.to(device), vt.to(device), tt.to(device), score.to(device), lambda_t.to(device), eps_xt.to(device), eps_score.to(device)

def get_sigma_t(observed_values, observed_times, t_fine, dynamics_kind, sigma=0.001, device=None):

    
    # keep everything on the same device to avoid CPU/GPU mismatches
    device = device or getattr(observed_values, "device", None)
    sigma_value = sigma.item() if torch.is_tensor(sigma) else float(sigma)

    if dynamics_kind=='ode' or dynamics_kind=='sde_constant_sigma':
        return torch.full((len(t_fine),), sigma_value, device=device)
    elif dynamics_kind == "sde_quadratic_sigma":
        t = torch.as_tensor(t_fine, device=device, dtype=torch.float32)              # (M,)
        ot = torch.as_tensor(observed_times, device=device, dtype=torch.float32)     # (K,)

        # interval index i for each t: ot[i] <= t < ot[i+1]
        idx = torch.bucketize(t, ot[1:], right=False)   # in [0, K-1]
        idx = idx.clamp(0, ot.numel() - 2)              # in [0, K-2]

        t_i = ot[idx]
        t_ip1 = ot[idx + 1]


        return sigma * torch.sqrt((t - t_i) * (t_ip1 - t))/(t_ip1 - t_i)
        

    else:
        raise NotImplementedError(f"Sigma scheme not implemented for dynamics_kind={dynamics_kind!r}")



def get_sigma_derivative_t(observed_values, observed_times, t_fine, dynamics_kind, interpolant_kind, sigma=0.001, device=None):

    device = device or getattr(observed_values, "device", None)

    if dynamics_kind=='ode':
        return torch.full((len(t_fine),), 0, device=device)
    elif dynamics_kind=='sde_constant_sigma':
        return torch.full((len(t_fine),), 0, device=device)
    
    elif dynamics_kind == "sde_quadratic_sigma":
        t = torch.as_tensor(t_fine, device=device, dtype=torch.float32)              # (M,)
        ot = torch.as_tensor(observed_times, device=device, dtype=torch.float32)     # (K,)

        idx = torch.bucketize(t, ot[1:], right=False).clamp(0, ot.numel() - 2)
        t_i = ot[idx]
        t_ip1 = ot[idx + 1]

        # d/dt [(t - t_i)(t_{i+1} - t)] = t_i + t_{i+1} - 2t
        term= torch.clamp(torch.sqrt((t - t_i) * (t_ip1 - t)), min=0.01)
        
        return sigma * (t_i + t_ip1 - 2.0 * t)/(2*(t_ip1 - t_i)*term)
    else:
        raise NotImplementedError(f"Sigma derivative not implemented for dynamics_kind={dynamics_kind!r}")

# def get_velocity_score_and_noise(der_mu_t, mu_t, observed_values, observed_times, t_fine, dynamics_kind, interpolant_kind, sigma=0.001):
#     # velocity:
#     # (der_simga_t/sigma_t)*(x-mu_t)+der_mu_t


#     sigma_t= get_sigma_t(observed_values, observed_times, t_fine, dynamics_kind, sigma=sigma)
#     der_sigma_t= get_sigma_derivative_t(observed_values, observed_times, t_fine, dynamics_kind, interpolant_kind, sigma=sigma)
#     eps_xt= torch.randn_like(mu_t)

#     if dynamics_kind=='ode':
#         velocity= der_mu_t
#     else:
#         # for gaussian paths x-mu_t= eps_x*sigma_t=> first term== der_sigma_t*eps_x
#         velocity= der_sigma_t*eps_xt+der_mu_t

#     # score

#     eps_score= torch.randn_like(mu_t)
#     score= -eps_score/sigma_t

#     xt_d= mu_t+sigma_t*eps_xt

#     if dynamics_kind=='ode':
#         lambda_t= 0*torch.ones_like(eps_score)
#     else:
#         lambda_t= (2/sigma**2)*sigma_t


#     return velocity, xt_d, score, eps_xt, eps_score, lambda_t


def get_velocity_score_and_noise(der_mu_t, mu_t, observed_values, observed_times, t_fine, dynamics_kind, interpolant_kind, sigma=0.001):
    # velocity:
    # (der_simga_t/sigma_t)*(x-mu_t)+der_mu_t

    
    sigma_t= get_sigma_t(observed_values, observed_times, t_fine, dynamics_kind, sigma=sigma, device=mu_t.device)
    der_sigma_t= get_sigma_derivative_t(observed_values, observed_times, t_fine, dynamics_kind, interpolant_kind, sigma=sigma, device=mu_t.device)
    sigma_value = sigma.item() if torch.is_tensor(sigma) else float(sigma)

    # for score
    if dynamics_kind=='ode':
        lambda_t= torch.full((len(mu_t),), 0, device=mu_t.device)
    else:
        lambda_t= (2/sigma_value**2)*sigma_t


    return sigma_t, lambda_t, der_sigma_t



import torch

@torch.no_grad()
def estimate_sigma_from_loader(
    loader,
    value_key: str = "values",
    time_key: str = "times",
    mask_key: str = "mask",
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
    per_dim: bool = False,
    eps_dt: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate additive diffusion sigma from trajectory increments:
        sigma^2 ~= (1/d) * sum ||dx||^2 / sum dt   (isotropic)
    or per-dimension:
        sigma_j^2 ~= sum dx_j^2 / sum dt

    Assumes x has shape (B, T, D).
    times: (B, T) or (B, T, 1). mask: (B, T, D) optional.
    Returns:
      - scalar tensor if per_dim=False
      - (D,) tensor if per_dim=True
    """
    num = None  # numerator accumulator
    den = torch.tensor(0.0, device=device)  # denominator accumulator (sum dt)

    batches_seen = 0
    for batch in loader:
        if max_batches is not None and batches_seen >= max_batches:
            break
        batches_seen += 1

        x = batch[value_key].to(device)              # (B,T,D)
        t = batch[time_key].to(device)               # (B,T) or (B,T,1)
        m = batch.get(mask_key, None)
        if m is not None:
            m = m.to(device).bool()                  # (B,T,D)

        if t.dim() == 3 and t.size(-1) == 1:
            t = t.squeeze(-1)                        # (B,T)

        # increments
        dx = x[:, 1:, :] - x[:, :-1, :]              # (B,T-1,D)
        dt = t[:, 1:] - t[:, :-1]                    # (B,T-1)

        # keep only positive dt
        valid_dt = dt > eps_dt                       # (B,T-1)
        dt = dt.clamp_min(eps_dt)

        if m is not None:
            # only use increments where BOTH endpoints are observed
            valid_mask = m[:, 1:, :] & m[:, :-1, :]  # (B,T-1,D)
            valid = valid_mask & valid_dt.unsqueeze(-1)
        else:
            valid = valid_dt.unsqueeze(-1).expand_as(dx)

        # accumulate denominator as sum dt over valid increments (per dim or shared)
        if per_dim:
            # sum dt per dimension = sum over (B,T-1) of dt where that dim valid
            den_dim = (dt.unsqueeze(-1) * valid.float()).sum(dim=(0, 1))  # (D,)
            dx2 = (dx.pow(2) * valid.float()).sum(dim=(0, 1))             # (D,)
            if num is None:
                num = dx2
                den = den_dim
            else:
                num = num + dx2
                den = den + den_dim
        else:
            # shared denom: sum dt over all valid increments (counted per dim)
            # numerator: sum ||dx||^2 over valid dims
            dx2_sum = (dx.pow(2) * valid.float()).sum()                   # scalar
            dt_sum  = (dt.unsqueeze(-1) * valid.float()).sum()            # scalar
            if num is None:
                num = dx2_sum
                den = dt_sum
            else:
                num = num + dx2_sum
                den = den + dt_sum

    if num is None or torch.any(den <= 0):
        raise RuntimeError("No valid increments found to estimate sigma.")

    sigma2 = num / den
    sigma = torch.sqrt(torch.clamp(sigma2, min=0.0))

    return sigma
