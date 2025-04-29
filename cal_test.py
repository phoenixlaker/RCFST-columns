# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/29 08:18
@Author  : Mjy
@Site    : 
@File    : cal_test.py
@Software: PyCharm
"""
import sympy as sp
import numpy as np
import math
import xgboost as xgb

def HLH_equ(H, B, t, e, L, Fy, Fck):
    # 参数初始化
    As = 2 * H * t + 2 * t * (B - 2 * t)
    Ac = (B - 2 * t) * (H - 2 * t)
    P = sp.symbols('P')

    # 计算ksi和alpha1
    ksi = (As * Fy) / (Ac * Fck)
    alpha1 = As / Ac
    Asc = As + Ac

    # 计算a2和b2
    if Fy <= 450:
        a2 = 1
        if Fck <= 41:
            b2 = 1
        else:
            b2 = (Fck / 41) ** 0.05 * (450 / Fy) ** 0.5
    else:
        a2 = (450 / Fy) ** 0.3
        b2 = (Fck / 41) ** 0.25 * (450 / Fy)

    # 计算B1, C1, fscy
    B1 = 0.1381 * (Fy / 235) ** a2 + 0.7646
    C1 = -0.0727 * (Fck / 20) ** b2 + 0.0216
    fscy = (1.212 + B1 * ksi + C1 * ksi ** 2) * Fck

    # 计算gamma_m, M0, P0
    gamma_m = 0.48 * math.log(ksi + 0.1) + 1.04
    Wsc = B * H ** 2 / 6
    M0 = gamma_m * fscy * Wsc / 1e6
    P0 = Asc * fscy / 1e3

    # 计算lamb, lamb_p, lamb_0
    lamb = 2 * math.sqrt(3) * L / H
    lamb_p = 1811 / math.sqrt(Fy)
    lamb_0 = math.pi / math.sqrt((220 * ksi + 450) / (0.85 * ksi + 1.18) / Fck)

    # 计算d1, e1, a1, b1, c1
    d1 = (13500 + 4810 * math.log(235 / Fy)) * (25 / (Fck + 5)) ** 0.3 + (alpha1 / 0.1) ** 0.05
    e1 = -d1 / (lamb_p + 35) ** 3
    a1 = (1 + (35 + 2 * lamb_p - lamb_0) * e1) / (lamb_p - lamb_0) ** 2
    b1 = e1 - 2 * a1 * lamb_p
    c1 = 1.0 - lamb_0 ** 2 * a1 - lamb_0 * b1

    # 计算稳定系数fi
    if lamb <= lamb_0:
        fi = 1
    elif lamb_0 < lamb <= lamb_p:
        fi = a1 * lamb ** 2 + b1 * lamb + c1
    else:
        fi = d1 / (lamb + 35) ** 2

    # 计算ksi_0和n0
    if ksi <= 0.4:
        n0 = 0.5 - 0.318 * ksi
    else:
        n0 = 0.1 + 0.13 * ksi ** (-0.81)
    ksi_0 = 1 + 0.14 * ksi ** (-1.3)

    # 计算PE
    fscp = (0.263 * (Fy / 235) + 0.365 * (20.318 / Fck) + 0.104) * fscy
    epsl_scp = 3.01e-6 * Fy
    Esc = fscp / epsl_scp
    PE = (math.pi ** 2 * Esc * Asc) / (1e3 * lamb ** 2)

    # 定义方程
    a = 1 - 2 * fi ** 2 * n0
    b = (1 - ksi_0) / (fi ** 3 * n0 ** 2)
    c = 2 * (ksi_0 - 1) / n0
    d = 1 - 0.25 * P / PE
    M = P * e / 1000

    # 解方程
    eq1 = (1 / fi) * (P / P0) + (a / d) * (M / M0) - 1
    eq2 = -b * (P / P0) ** 2 - c * (P / P0) + (M / (d * M0)) - 1

    solutions1 = sp.solve(eq1, P)
    solutions2 = sp.solve(eq2, P)

    cof1 = [sol / P0 for sol in solutions1]
    cof2 = [sol / P0 for sol in solutions2]
    cof = 2 * fi ** 3 * n0

    # 筛选有效解
    N_u = None
    for sol in cof1:
        if 0 < sol < 1 and sol >= cof:
            N_u = sol * P0
            break

    if N_u is None:
        for sol in cof2:
            if 0 < sol < 1 and sol < cof:
                N_u = sol * P0
                break


    return N_u

mean = np.array([[1.56698319e+02],
                 [1.52984314e+02],
                 [4.64090588e+00],
                 [4.04520980e+02],
                 [5.73159384e+01],
                 [1.62416725e+03],
                 [4.39894678e+01],
                 [2.74872693e-01],
                 [4.01522258e+01],
                 [1.19440886e+03],
                 [1.25600616e+03],
                 [1.44877602e-01],
                 [3.00115173e+06],
                 [2.18227651e+06],
                 [5.96313977e+04],
                 [1.20540928e+03]])

std = np.array([[4.60558971e+01],
                [4.70302737e+01],
                [1.69857145e+00],
                [1.52380498e+02],
                [2.91625558e+01],
                [1.06282473e+03],
                [4.14640318e+01],
                [2.06793648e-01],
                [3.05838350e+01],
                [9.61442056e+02],
                [9.71773609e+02],
                [6.55761775e-02],
                [4.14682376e+06],
                [4.09302294e+06],
                [1.07828038e+05],
                [8.80251449e+02]])

booster = xgb.Booster()
booster.load_model('XGB.bin')
XGB = xgb.XGBRegressor()
XGB._Booster = booster


def inputfeature_generation(H, B, t, e, L, fy, fc, fck, Es, Ec):
    N_HLH = HLH_equ(H, B, t, e, L, fy, fck)
    As = 2 * H * t + 2 * t * (B - 2 * t)
    Ac = (B - 2 * t) * (H - 2 * t)
    Ic = (B - 2 * t) * (H - 2 * t) ** 3 / 12
    Is = B * H ** 3 / 12 - (B - 2 * t) * (H - 2 * t) ** 3 / 12
    e_H = e / H
    lamba = 2 * np.sqrt(3) * L / H
    N_steel = fy * As / 1000
    N_concrete = fc * Ac / 1000
    alpha = As / Ac
    EsIs = Es * Is / 1000000
    EcIc = Ec * Ic / 1000000
    Ncr = math.pi ** 2 * (EsIs + 0.6 * EcIc) / (L / 1000) ** 2 / 1000
    H_s = (H - mean[0, 0]) / std[0, 0]
    B_s = (B - mean[1, 0]) / std[1, 0]
    t_s = (t - mean[2, 0]) / std[2, 0]
    fy_s = (fy - mean[3, 0]) / std[3, 0]
    fc_s = (fc - mean[4, 0]) / std[4, 0]
    L_s = (L - mean[5, 0]) / std[5, 0]
    e_s = (e - mean[6, 0]) / std[6, 0]
    eH_s = (e_H - mean[7, 0]) / std[7, 0]
    lamba_s = (lamba - mean[8, 0]) / std[8, 0]
    Ns_s = (N_steel - mean[9, 0]) / std[9, 0]
    Nc_s = (N_concrete - mean[10, 0]) / std[10, 0]
    alpha_s = (alpha - mean[11, 0]) / std[11, 0]
    EsIs_s = (EsIs - mean[12, 0]) / std[12, 0]
    EcIc_s = (EcIc - mean[13, 0]) / std[13, 0]
    Ncr_s = (Ncr - mean[14, 0]) / std[14, 0]
    HLH_s = (N_HLH - mean[15, 0]) / std[15, 0]
    inp = np.array([H_s, B_s, t_s, fy_s, fc_s, L_s, e_s, eH_s, lamba_s,
                    Ns_s, Nc_s, alpha_s, EsIs_s, EcIc_s, Ncr_s, HLH_s])
    y_low = (N_HLH + 3.7517 * H_s - 31.7064 * B_s + 35.0404 * t_s - 142.3712 * fy_s + 42.5868 * fc_s + 16.2992 * L_s +
             9.7718 * e_s - 95.9316 * eH_s - 51.2813 * lamba_s + 207.5104 * Ns_s + 168.7317 * Nc_s - 1.5542 * alpha_s -
             38.3145 * EsIs_s - 92.9113 * EcIc_s - 60.3310 * Ncr_s - 85.0641 * HLH_s + 149.6561)
    inp = inp.reshape(1, 16)


    return y_low, inp


def final_out(H, B, t, e, L, fy, fc, fck, Es, Ec):
    y_low, inp = inputfeature_generation(H, B, t, e, L, fy, fc, fck, Es, Ec)
    y_pred_XGB = XGB.predict(inp)
    N_u = y_low + y_pred_XGB
    print(N_u)
    return N_u

final_out(H=100, B=100, t=10, e=10, L=100, fy=250, fc=25, fck=20, Es=200, Ec=200)