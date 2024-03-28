import torch

TRANSFORM_EPS = 1e-6


def axisangle2mat_torch(axisangle: torch.Tensor) -> torch.Tensor:
    theta2 = axisangle[:, :3].pow(2).sum(-1)
    small_angle = theta2 <= TRANSFORM_EPS
    theta = torch.clamp(theta2, min=TRANSFORM_EPS).sqrt()
    ang_x = axisangle[:, 0] / theta
    ang_y = axisangle[:, 1] / theta
    ang_z = axisangle[:, 2] / theta
    s = torch.sin(theta)
    c = torch.cos(theta)
    o_c = 1 - c
    # large angle
    mat1 = torch.stack(
        (
            c + ang_x * ang_x * o_c,
            ang_x * ang_y * o_c - ang_z * s,
            ang_y * s + ang_x * ang_z * o_c,
            ang_z * s + ang_x * ang_y * o_c,
            c + ang_y * ang_y * o_c,
            -ang_x * s + ang_y * ang_z * o_c,
            -ang_y * s + ang_x * ang_z * o_c,
            ang_x * s + ang_y * ang_z * o_c,
            c + ang_z * ang_z * o_c,
        ),
        -1,
    )
    # small angle
    ones = torch.ones_like(o_c)
    mat2 = torch.stack(
        (
            ones,
            -axisangle[:, 2],
            axisangle[:, 1],
            axisangle[:, 2],
            ones,
            -axisangle[:, 0],
            -axisangle[:, 1],
            axisangle[:, 0],
            ones,
        ),
        -1,
    )
    # output
    mat = torch.where(small_angle[..., None], mat2, mat1).reshape((-1, 3, 3))
    mat = torch.cat((mat, axisangle[:, -3:, None]), -1)

    return mat


def mat2axisangle_torch(mat: torch.Tensor) -> torch.Tensor:
    r00 = mat[:, 0, 0]
    r01 = mat[:, 0, 1]
    r02 = mat[:, 0, 2]
    r10 = mat[:, 1, 0]
    r11 = mat[:, 1, 1]
    r12 = mat[:, 1, 2]
    r20 = mat[:, 2, 0]
    r21 = mat[:, 2, 1]
    r22 = mat[:, 2, 2]

    mask_d2 = r22 < TRANSFORM_EPS
    mask_d0_d1 = r00 > r11
    mask_d0_nd1 = r00 < -r11

    # case 1
    s1 = 2 * torch.sqrt(torch.clamp(r00 + r11 + r22 + 1, min=TRANSFORM_EPS))
    w1 = s1 / 4
    x1 = (r21 - r12) / s1
    y1 = (r02 - r20) / s1
    z1 = (r10 - r01) / s1

    # case 2
    s2 = 2 * torch.sqrt(torch.clamp(r00 - r11 - r22 + 1, min=TRANSFORM_EPS))
    w2 = (r21 - r12) / s2
    x2 = s2 / 4
    y2 = (r01 + r10) / s2
    z2 = (r02 + r20) / s2

    # case 3
    s3 = 2 * torch.sqrt(torch.clamp(r11 - r00 - r22 + 1, min=TRANSFORM_EPS))
    w3 = (r02 - r20) / s3
    x3 = (r01 + r10) / s3
    y3 = s3 / 4
    z3 = (r12 + r21) / s3

    # case 4
    s4 = 2 * torch.sqrt(torch.clamp(r22 - r00 - r11 + 1, min=TRANSFORM_EPS))
    w4 = (r10 - r01) / s4
    x4 = (r02 + r20) / s4
    y4 = (r12 + r21) / s4
    z4 = s4 / 4

    case1 = (~mask_d2) & ~(mask_d0_nd1)
    case2 = mask_d2 & mask_d0_d1
    case3 = mask_d2 & (~mask_d0_d1)

    w = torch.where(case1, w1, torch.where(case2, w2, torch.where(case3, w3, w4)))
    x = torch.where(case1, x1, torch.where(case2, x2, torch.where(case3, x3, x4)))
    y = torch.where(case1, y1, torch.where(case2, y2, torch.where(case3, y3, y4)))
    z = torch.where(case1, z1, torch.where(case2, z2, torch.where(case3, z3, z4)))

    neg_w = w < 0
    x = torch.where(neg_w, -x, x)
    y = torch.where(neg_w, -y, y)
    z = torch.where(neg_w, -z, z)
    w = torch.where(neg_w, -w, w)

    tmp = x * x + y * y + z * z
    si = torch.sqrt(torch.clamp(tmp, min=TRANSFORM_EPS))
    theta = 2 * torch.atan2(si, w)
    fac = torch.where(tmp > TRANSFORM_EPS, theta / si, 2.0 / w)

    x = x * fac
    y = y * fac
    z = z * fac

    axisangle = torch.cat((x[:, None], y[:, None], z[:, None], mat[:, :, -1]), -1)
    return axisangle
