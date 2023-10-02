import torch
from torch import autograd
import numpy as np

def lennard_jones_potential(positions, epsilon, sigma, w):
    num_particles = positions.shape[0]
    LJ_potential = torch.tensor(0.0)

    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            r = torch.norm(positions[j] - positions[i])
            sigma_sq = sigma ** 2
            r_sq = r ** 2
            sr2 = sigma_sq / r_sq
            sr6 = sr2 ** 3
            sr12 = sr6 ** 2
            LJ_potential += 4 * epsilon * (sr12 - sr6) - w * sr6

    return LJ_potential

def velocity_verlet(positions, start, stop, dt, epsilon, sigma, w):
    steps = int((stop - start) / dt)

    vv_positions = positions.clone()
    vv_velocities = torch.zeros_like(positions)
    a = torch.zeros_like(positions)

    for t in range(steps):
        LJ_potential = lennard_jones_potential(vv_positions, epsilon, sigma, w)
        a = autograd.grad(outputs=LJ_potential, inputs=vv_positions, create_graph=True)[0]
        half_step_velocities = vv_velocities + 0.5 * dt * a
        vv_positions += half_step_velocities * dt
        LJ_potential = lennard_jones_potential(vv_positions, epsilon, sigma, w)
        a = autograd.grad(outputs=LJ_potential, inputs=vv_positions, create_graph=True)[0]
        vv_velocities = half_step_velocities + 0.5 * dt * a

    return vv_positions

def radial_distribution_function(positions, density, dr, start, stop, dt, epsilon, sigma, w):
    def rdf(positions):
        num_particles = positions.shape[0]
        r = torch.zeros(num_particles * (num_particles - 1) // 2)

        r_max = 0.0
        cal = 0
        for i in range(num_particles - 1):
            for j in range(i + 1, num_particles):
                r[cal] = torch.norm(positions[j] - positions[i])
                if r[cal] >= r_max:
                    r_max = r[cal]
                cal += 1

        layer_num = int(r_max / dr)
        n = torch.zeros(layer_num + 1)

        for r_cal in r:
            if (r_cal < r_max):
                layer = int(r_cal / dr)
                n[layer] += 1

        rdf_values = torch.zeros(layer_num)

        for ii in range(0, layer_num):
            r_val = ii * dr
            volume = 4 * np.pi * r_val ** 2 * dr

            if (volume > 0):
                rdf_values[ii] = n[ii] / (density * num_particles * volume)
            else:
                rdf_values[ii] = 0

        return rdf_values

    rdf_positions = velocity_verlet(positions, start, stop, dt, epsilon, sigma, w)
    rdf_result = rdf(rdf_positions)

    return rdf_result

if __name__ == '__main__':
    # This program completes the modification of epsilon, sigma, and w using backpropagation.
    # The loss is the radial distribution function.

    # Target positions
    initial_positions = torch.tensor([
        [0.5003, 0.6019, 0.8131], [0.8531, 0.6735, 0.0335], [0.0340, 0.7145, 0.4484], [0.8632, 0.2229, 0.4975],
        [0.8591, 0.2504, 0.1940], [0.0264, 0.6417, 0.7706], [0.6427, 0.6628, 0.3692], [0.4019, 0.1254, 0.3932],
        [0.2850, 0.2754, 0.3543], [0.9691, 0.7344, 0.3987], [0.2232, 0.0399, 0.5536], [0.3800, 0.8151, 0.1721],
        [0.6674, 0.4491, 0.8612], [0.6625, 0.2915, 0.6576], [0.8244, 0.9042, 0.4357], [0.0539, 0.9775, 0.1198],
        [0.1270, 0.3761, 0.9669], [0.2301, 0.7711, 0.4011], [0.8430, 0.0778, 0.1193], [0.0000, 0.0000, 0.0000]],
        requires_grad=True)

    start = 0
    stop = 1e-5
    dt = 1e-7

    epsilon = torch.tensor(0.6, requires_grad=True)
    sigma = torch.tensor(0.6, requires_grad=True)
    w = torch.tensor(1.0, requires_grad=True)

    # Radial distribution function parameters
    density = 1.0
    dr = 0.1

    target_rdf = radial_distribution_function(initial_positions, density, dr,
                                              start, stop, dt, epsilon, sigma, w)

    # simulation positions
    epsilon = torch.tensor(0.55, requires_grad=True)
    sigma = torch.tensor(0.55, requires_grad=True)
    w = torch.tensor(1.1, requires_grad=True)

    # Back propagation
    optimizer = torch.optim.SGD([epsilon, sigma, w], lr=0.01)
    BP_steps = 0

    while BP_steps <= 100:
        simulation_rdf = radial_distribution_function(initial_positions, density, dr,
                                                      start, stop, dt, epsilon, sigma, w)

        rdf_loss = torch.sum((simulation_rdf - target_rdf) ** 2)
        optimizer.zero_grad()
        rdf_loss.backward()

        if BP_steps % 10 == 0:
            print(f"Step {BP_steps}: RDF Loss = {rdf_loss.item()}")

        optimizer.step()
        BP_steps += 1

        if rdf_loss.item() < 1e-6:
            break

    print(f"Epsilon: {epsilon.item()}")
    print(f"Sigma: {sigma.item()}")
    print(f"W: {w.item()}")
    print(f"Backpropagation Steps: {BP_steps}")