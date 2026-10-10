#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""SN1993J: single-file TransFit 11-parameter candidate + dynamic nested sampling, for PyCharm.

在 PyCharm 中直接 Run 此文件。不需要命令行参数、其他.py、JSON配置或滤光片文件。
数据读取下方 DATA_FILE 指定的 CSV。仅需第三方包：numpy scipy numba dynesty>=3.0,<4。
已验证解释器：/opt/anaconda3/bin/python3 (Python 3.13)。建议 Python 3.11--3.13。
安装依赖（PyCharm Terminal）：python -m pip install numpy scipy numba "dynesty>=3.0,<4"

原9参数 + 包层质量比例 + 有效γ逃逸参数。R0和n自由，允许热能贴上限，不画图。
初始R0约300Rsun只是优化起点，不保证新采样仍给出300Rsun；短链不证明收敛。
这是实验性结构扩展，不是官方发布的完整IIb水动力模型。
上游作者：Liangduan Liu、Yuhao Zhang；TransFit 0.2.0。
来源 https://github.com/CCNUastro/TransFit
记录commit 4e03b2d35fc90ac47d9c200aed0872b3c453d7d9。
文件末尾包含GPL-3.0许可证全文；修改部分同样按GPL-3.0-only分发。
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent

# === USER SETTINGS BEGIN ===
# 一般只改这一段。请使用 PyCharm 的 Run，不要粘贴到 Python Console。
RUN_MODE = 'nested'   # 'nested', 'evaluate', 'fit', 'predict'
DATA_FILE = r"/Users/weina/Documents/A/doubl_peak/model/SN 1993J/SN1993j.csv"
RESULT_DIR = str(ROOT / 'SN1993J_Ek_tshift_dynesty_results')
N_WORKERS = 4           # 可改为4；PyCharm Run支持spawn多进程
RESUME = False         # 仅恢复本版本checkpoint.pkl，不支持旧emcee链
RANDOM_SEED = 1993
NLIVE_INIT = 500
NLIVE_BATCH = 250
DLOGZ_INIT = 0.05       # 初始嵌套阶段的证据停止阈值
N_EFFECTIVE = 10000     # 动态阶段目标有效后验样本数；不是步数
MAXCALL = None         # 可限制总调用数；达到限制不代表收敛
MAXITER = None
MAXBATCH = None
CHECKPOINT_EVERY = 60   # 秒；完成一次迭代后才检查保存间隔
# velocity_uniform 保持旧版Mej、v独立均匀先验的变换；Ek仍是模型自由参数。
# energy_uniform 对Ek独立均匀，使用下方Ek范围；这会改变先验。
ENERGY_PRIOR = 'velocity_uniform'
VELOCITY_PRIOR_BOUNDS = [0.2, 2.5]  # 仅velocity_uniform使用，单位1e9 cm/s
EK_BOUNDS = [0.01, 50.0]            # 仅energy_uniform使用，单位1e51 erg
FIT_STARTS = 3
FIT_MAX_NFEV = 200
CHECK_FINER_GRID = True
PARAMETER_FILE = None   # evaluate/fit/predict使用；必须是本版本参数JSON
PREDICTION_POINTS = 400

# 模型设置：不改变它们也能复现SN1993J候选结果。
# dynesty通过prior_transform实现先验；log_likelihood中不重复添加先验或雅可比。
MODEL_CONFIG = {'name': 'SN1993J',
 'data_file': DATA_FILE,
 'time_column': 'MJD',
 'time_format': 'JD',
 'reference_epoch': 2449074.0,
 'bands': ['B', 'V', 'R', 'I'],
 'phase_min_days': 0.0,
 'phase_max_days': 60.0,
 'zero_is_missing': True,
 'distance_mpc': 3.63,
 'redshift': 0.0,
 'ebv': 0.19,
 'rv': 3.1,
 'data_already_dereddened': False,
 'sigma_systematic_mag': 0.0,
 'kappa': 0.07,
 'q_core': 0.3333333333333333,
 'nx': 100,
 'ny': 3000,
 't_max_days': 65.0,
 'output_step_days': 1e-05,
 'prior': ENERGY_PRIOR,
 'velocity_prior_bounds': VELOCITY_PRIOR_BOUNDS,
 'bounds': {'M_ej_Msun': [0.3, 8.0],
            'E_k_1e51_erg': EK_BOUNDS,
            'E_th_1e49_erg': [0.01, 100.0],
            'M_Ni_Msun': [0.01, 0.5],
            'R0_Rsun': [30.0, 1000.0],
            'Ni_mass_fraction_in_mixed_region': [0.001, 1.0],
            'f_Ni': [0.001, 1.0],
            'T_floor_K': [3000.0, 8000.0],
            'kappa_gamma_cm2_g': [0.005, 0.2],
            'tshift_days': [-5.0, 2.69],
            'f_env': [0.001, 0.3],
            'n': [5.1, 14.0]}}
# 数据路径统一取上面的 DATA_FILE，避免修改两处。
MODEL_CONFIG['data_file'] = DATA_FILE

# 内部坐标：[logMej,logEk,logEth,logMNi,logR0,logNi丰度,logTfloor,
#           logkappa_gamma,爆炸时刻偏移,logf_env,n]；log均为log10。
INITIAL_COORDINATES = [0.1579854315224721, -0.37708001358625215, 1.99905346571335, -0.8741990931479422, 2.476779673501281, -0.0015460376253720908, 3.694417966103678, -1.9290531182272461, 0.9940657433058215, -0.9347769950413347, 6.344536851980787]
# === USER SETTINGS END ===

import sys,json,hashlib,importlib.metadata,time,multiprocessing as mp,contextlib,csv,copy
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import numba as nb
import dynesty
from dynesty import utils as dyfunc
from scipy.special import logsumexp
from scipy.optimize import least_squares

# === Embedded physical constants and upstream definitions ===
# ---- math
PI: float = float(np.pi)

# ---- cgs physical constants
C_LIGHT: float = 2.99792458e10          # cm/s
H_PLANCK: float = 6.62607015e-27        # erg*s
K_BOLTZ: float = 1.380649e-16           # erg/K
SIGMA_SB: float = 5.670316237574177e-5  # erg/s/cm^2/K^4
SIGMA_T: float = 6.652e-25              # cm^2
M_P: float = 1.673e-24                  # g
M_E: float = 9.109e-28                  # g
Q_E: float = 4.803e-10                  # statcoulomb

# ---- astronomy (cgs)
M_SUN: float = 1.9889999e33             # g
R_SUN: float = 6.96e10                  # cm
L_SUN: float = 3.828e33                 # erg/s
PC: float = 3.086e18                    # cm
MPC: float = 1.0e6 * PC                 # cm

# ---- time / frequency / flux density helpers
KM: float = 1.0e5                       # cm/s
DAY: float = 86400.0                    # s
YR: float = 365.0 * DAY                 # s (project convention)
GHZ: float = 1.0e9                      # Hz
UJY: float = 1.0e-29                    # erg/s/cm^2/Hz
MJY: float = 1.0e-26                    # erg/s/cm^2/Hz
JY: float = 1.0e-23                     # erg/s/cm^2/Hz

# ---- (optional) cosmology defaults (only if you truly want "project default")
H0: float = 71.0 * KM / MPC
OMEGA_M: float = 0.27
OMEGA_L: float = 0.73

# ---- Ni/Co heating (project model convention)
EPSILON_NI: float = 3.9e10              # erg/s/g
EPSILON_CO: float = 6.8e9               # erg/s/g
TAU_NI: float = 8.8 * DAY               # s
TAU_CO: float = 111.3 * DAY             # s
EPSILON_RATIO: float = EPSILON_CO / (EPSILON_NI - EPSILON_CO)

@dataclass(frozen=True)
class NickelTransportState:
    """Solved bolometric transport state for the Nickel model.

    Uniform density preserves the historical fixed-outer-boundary solve and
    homologous effective blackbody.  BPL and exponential density use the
    physical ``tau=2/3`` diffusion boundary; power deposited outside that
    domain is carried by ``Ldirect`` and never enters the diffusion matrix.
    """

    t_s: np.ndarray
    Lbol: np.ndarray
    Lphotospheric: np.ndarray
    Ldirect: np.ndarray
    q_ph: np.ndarray
    Rph: np.ndarray
    Tph: np.ndarray
    Rhom: np.ndarray
    photosphere_valid: np.ndarray
    density_profile: str


def _gamma_deposition_fraction(t_phys, t_gamma):
    """Return ``1 - exp[-(t_gamma/t)^2]`` with its ``t=0`` limit."""
    t_phys = np.asarray(t_phys, dtype=float)
    deposition = np.ones_like(t_phys)
    positive_time = t_phys > 0.0
    deposition[positive_time] = 1.0 - np.exp(
        -(float(t_gamma) / t_phys[positive_time]) ** 2
    )
    return deposition


def _radioactive_heating_shape(t_phys, t_gamma):
    """Return deposited Ni-to-Co heating normalized by ``epsilon_Ni-epsilon_Co``."""
    t_phys = np.asarray(t_phys, dtype=float)
    e_co_ratio = EPSILON_CO / (EPSILON_NI - EPSILON_CO)
    radioactive = (
        np.exp(-t_phys / TAU_NI)
        + e_co_ratio * np.exp(-t_phys / TAU_CO)
    )
    return radioactive * _gamma_deposition_fraction(t_phys, t_gamma)

@dataclass(frozen=True)
class BlackbodySED:
    """
    Blackbody SED: L_nu = 4*pi^2*R^2*B_nu(T).
    """
    Tmin: float = 1.0   # K, avoid divide-by-zero
    Rmin: float = 1.0   # cm

    def planck_Bnu(self, nu_hz: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """
        Planck function B_nu [erg/s/cm^2/Hz/sr].
        Vectorized: nu (Nb,) and T (Nt,) -> (Nb, Nt)
        """
        nu = np.asarray(nu_hz, dtype=float).reshape(-1, 1)   # (Nb,1)
        T = np.asarray(T_K, dtype=float).reshape(1, -1)      # (1,Nt)
        T = np.clip(T, self.Tmin, np.inf)

        x = (H_PLANCK * nu) / (K_BOLTZ * T)  # (Nb,Nt)
        # In the Wien tail exp(x) may overflow, which correctly corresponds
        # to B_nu underflowing to zero rather than a numerical failure.
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            denom = np.expm1(x)
            denom = np.where(denom == 0.0, np.inf, denom)
            Bnu = (2.0 * H_PLANCK * nu**3) / (C_LIGHT**2) / denom
        return np.where(np.isfinite(Bnu), Bnu, np.nan)

    def lnu(self, nu_rest_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray) -> np.ndarray:
        """
        Luminosity density L_nu [erg/s/Hz], shape (Nb, Nt).
        """
        R = np.asarray(R_cm, dtype=float).reshape(1, -1)
        R = np.clip(R, self.Rmin, np.inf)

        Bnu = self.planck_Bnu(nu_rest_hz, Teff_K)   # (Nb,Nt)
        Lnu = (4.0 * PI * PI) * (R**2) * Bnu
        return np.where(np.isfinite(Lnu), Lnu, np.nan)

    def fnu(
        self,
        nu_obs_hz: np.ndarray,
        Teff_K: np.ndarray,
        R_cm: np.ndarray,
        DL_cm: float,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Observer-frame flux density F_nu [erg/s/cm^2/Hz], shape (Nb, Nt).

        Uses the luminosity-distance convention:
          nu_rest = nu_obs * (1+z)
          Fnu = (1+z) * Lnu / (4*pi*DL^2)
        """
        zp1 = 1.0 + (z or 0.0)
        nu_rest = np.asarray(nu_obs_hz, dtype=float) * zp1
        Lnu = self.lnu(nu_rest, Teff_K, R_cm)
        Fnu = (zp1 * Lnu) / (4.0 * PI * (DL_cm**2))
        return np.where(Fnu > 0.0, Fnu, np.nan)

    def abmag(
        self,
        nu_obs_hz: np.ndarray,
        Teff_K: np.ndarray,
        R_cm: np.ndarray,
        DL_cm: float,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Legacy convenience wrapper.

        The main code path now keeps the model core in F_nu and performs
        magnitude conversion in `transfit.modules.magnitudes`.
        """
        Fnu = self.fnu(nu_obs_hz, Teff_K, R_cm, DL_cm=DL_cm, z=z)
        mab = -2.5 * np.log10(Fnu) - 48.6
        return mab

def _as_lambda_um_array(lambda_um: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(lambda_um, float)
    if arr.size == 0:
        raise ValueError("lambda_um must be non-empty.")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("lambda_um must contain positive finite values.")
    x = 1.0 / arr
    if np.any((x < 0.3) | (x > 10.0)):
        raise ValueError(
            "Extinction laws currently support wavelengths in the range "
            "0.1-3.33 micron (x = 0.3-10 micron^-1)."
        )
    return arr


def _ccm89_ab(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros_like(x, dtype=float)
    b = np.zeros_like(x, dtype=float)

    ir = (x >= 0.3) & (x < 1.1)
    if np.any(ir):
        xx = x[ir]
        a[ir] = 0.574 * xx ** 1.61
        b[ir] = -0.527 * xx ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    if np.any(opt):
        y = x[opt] - 1.82
        a[opt] = (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b[opt] = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )

    uv = (x >= 3.3) & (x <= 8.0)
    if np.any(uv):
        xx = x[uv]
        fa = np.zeros_like(xx, dtype=float)
        fb = np.zeros_like(xx, dtype=float)
        mask = xx >= 5.9
        if np.any(mask):
            y = xx[mask] - 5.9
            fa[mask] = -0.04473 * y**2 - 0.009779 * y**3
            fb[mask] = 0.2130 * y**2 + 0.1207 * y**3
        a[uv] = 1.752 - 0.316 * xx - 0.104 / ((xx - 4.67) ** 2 + 0.341) + fa
        b[uv] = -3.090 + 1.825 * xx + 1.206 / ((xx - 4.62) ** 2 + 0.263) + fb

    fuv = (x > 8.0) & (x <= 10.0)
    if np.any(fuv):
        y = x[fuv] - 8.0
        a[fuv] = -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3
        b[fuv] = 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3

    return a, b


def extinction_axav(lambda_um,*,law,rv):
    if law!='ccm89':raise ValueError('This single-file version supplies CCM89 extinction only')
    if not np.isfinite(rv) or rv<=0:raise ValueError('Invalid RV')
    arr=_as_lambda_um_array(lambda_um);a,b=_ccm89_ab(1./arr)
    return np.asarray(a+b/rv,float)

class _NonPhysicalModelOutput(ValueError):
    pass

def _nickel_two_component_blackbody(transport, T_floor: float):
    """Return diluted photospheric and optically thin blackbody components.

    The physical photosphere uses ``Lphotospheric`` and ``Rph``.  When its
    effective temperature falls below ``T_floor``, the color temperature is
    fixed at the floor and the blackbody area is diluted so its frequency
    integral remains exactly ``Lphotospheric``.  ``Ldirect`` is normalized as
    a separate floor-temperature continuum.  The two components therefore
    integrate to ``Lbol`` without assigning optically thin power to ``Rph``.
    """
    floor = float(T_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("T_floor must be finite and > 0 for photospheric emission.")
    luminosity = np.asarray(transport.Lbol, float)
    photospheric_luminosity = np.asarray(transport.Lphotospheric, float)
    direct_luminosity = np.asarray(transport.Ldirect, float)
    radius_physical = np.asarray(transport.Rph, float)
    photosphere_valid = np.asarray(transport.photosphere_valid, bool)
    if not (
        luminosity.shape
        == photospheric_luminosity.shape
        == direct_luminosity.shape
        == radius_physical.shape
        == photosphere_valid.shape
    ):
        raise _NonPhysicalModelOutput(
            "Nickel Lbol, Lphotospheric, Ldirect, Rph, and photosphere_valid "
            "must have the same shape."
        )
    if (
        np.any(~np.isfinite(photospheric_luminosity))
        or np.any(photospheric_luminosity < 0.0)
        or np.any(~np.isfinite(direct_luminosity))
        or np.any(direct_luminosity < 0.0)
    ):
        raise _NonPhysicalModelOutput(
            "Nickel Lphotospheric and Ldirect must be finite and non-negative."
        )
    if not np.allclose(
        luminosity,
        photospheric_luminosity + direct_luminosity,
        rtol=5.0e-12,
        atol=0.0,
    ):
        raise _NonPhysicalModelOutput(
            "Nickel transport must satisfy Lbol=Lphotospheric+Ldirect."
        )

    photo_active = photosphere_valid & (photospheric_luminosity > 0.0)
    invalid_photo_radius = ~np.isfinite(radius_physical) | (radius_physical <= 0.0)
    if np.any(photo_active & invalid_photo_radius):
        raise _NonPhysicalModelOutput(
            "A luminous Nickel photospheric component requires finite positive Rph."
        )

    temperature_photo = np.full(luminosity.shape, floor, dtype=float)
    radius_photo = np.zeros(luminosity.shape, dtype=float)
    if np.any(photo_active):
        temperature_effective = (
            photospheric_luminosity[photo_active]
            / (4.0 * PI * SIGMA_SB * radius_physical[photo_active] ** 2)
        ) ** 0.25
        temperature_photo[photo_active] = np.maximum(temperature_effective, floor)
        # This effective area is Rph**2 times the usual dilution factor
        # (T_eff/T_color)**4.  Above the floor it is exactly the physical Rph.
        radius_photo[photo_active] = np.sqrt(
            photospheric_luminosity[photo_active]
            / (4.0 * PI * SIGMA_SB * temperature_photo[photo_active] ** 4)
        )

    temperature_direct = np.full(luminosity.shape, floor, dtype=float)
    radius_direct = np.sqrt(
        direct_luminosity / (4.0 * PI * SIGMA_SB * floor**4)
    )
    return temperature_photo, radius_photo, temperature_direct, radius_direct


# === Corrected transport ===
@nb.njit(cache=True)
def integ(lo,hi,power):
 if hi<=lo:return 0.
 if abs(power+1)<1e-10:return np.log1p((hi-lo)/lo)
 if hi-lo<1e-3*lo:return lo**(power+1)*np.expm1((power+1)*np.log1p((hi-lo)/lo))/(power+1)
 return (hi**(power+1)-lo**(power+1))/(power+1)


@nb.njit(cache=True)
def moment(lo,hi,order,qc,outer_amp,outer_n):
 val=0.
 if lo<qc:val+=integ(lo,min(hi,qc),order)
 if hi>qc:val+=outer_amp*qc**outer_n*integ(max(lo,qc),hi,order-outer_n)
 return val


@nb.njit(cache=True)
def eta(q,qc,amp,n):return 1. if q<=qc else amp*(q/qc)**(-n)


@nb.njit(cache=True)
def geometry(faces,qph,qni,xi,qc,amp,n):
 nx=len(faces)-1;vol=np.zeros(nx);rho=np.ones(nx);cent=np.zeros(nx);src=np.zeros(nx)
 active=0
 for i in range(nx):
  lo=faces[i];hi=min(faces[i+1],qph)
  if hi-lo<1e-14:break
  active+=1;vol[i]=(hi-lo)*(hi*hi+hi*lo+lo*lo)/3
  rho[i]=moment(lo,hi,2,qc,amp,n)/vol[i]
  cent[i]=.75*(hi+lo)*(hi*hi+lo*lo)/(hi*hi+hi*lo+lo*lo)
  src[i]=xi*moment(lo,min(hi,qni),2,qc,amp,n)/vol[i]
 return vol,rho,cent,src,active


@nb.njit(cache=True)
def find_photo(faces,kr,target,qc,amp,n):
 col=0.
 for i in range(len(kr)-1,-1,-1):
  lo=faces[i];hi=faces[i+1];this=kr[i]*moment(lo,hi,0,qc,amp,n)
  if col+this>=target:
   left=lo;right=hi
   for j in range(38):
    mid=(left+right)/2
    if col+kr[i]*moment(mid,hi,0,qc,amp,n)>target:left=mid
    else:right=mid
   return (left+right)/2
  col+=this
 return faces[0]


@nb.njit(cache=True)
def solve_kernel(faces,t,heat,rho0,R0,vmax,kref,u0,tdiff,L0,Eth_dim,qni,xi,qc,amp,n,kind,fheat,Trec,floor_ratio,exact_resistance):
 nx=len(faces)-1;nt=len(t);qmin=faces[0];ar=4*SIGMA_SB/C_LIGHT
 vfull=(faces[1:]**3-faces[:-1]**3)/3;cent=(faces[:-1]+faces[1:])/2
 initial=np.zeros(nx)
 for i in range(nx):
  lo=faces[i];hi=faces[i+1]
  if kind==1:
   w=(1-fheat)*integ(lo,min(hi,qc),1)/integ(qmin,qc,1)+fheat*integ(max(lo,qc),hi,1)/integ(qc,1.,1)
  elif kind==3 or kind==4:
   w=moment(lo,hi,4,qc,amp,n)/moment(qmin,1.,4,qc,amp,n)
  else:w=integ(lo,hi,1)/integ(qmin,1.,1)
  initial[i]=Eth_dim*w/vfull[i]
 kr=np.ones(nx)
 if kind==2:
  for i in range(nx):kr[i]=.34/kref if cent[i]>=qc else 1.
 tauscale=kref*rho0*R0
 qold=find_photo(faces,kr,(2/3)/tauscale,qc,amp,n)
 volold,_,_,_,_=geometry(faces,qold,qni,xi,qc,amp,n)
 e=initial.copy()
 for i in range(nx):
  if volold[i]<=0:e[i]=0.
  else:
   hi=min(faces[i+1],qold);lo=faces[i]
   if kind==1:w=(1-fheat)*integ(lo,min(hi,qc),1)/integ(qmin,qc,1)+fheat*integ(max(lo,qc),hi,1)/integ(qc,1.,1)
   elif kind==3 or kind==4:w=moment(lo,hi,4,qc,amp,n)/moment(qmin,1.,4,qc,amp,n)
   else:w=integ(lo,hi,1)/integ(qmin,1.,1)
   e[i]=Eth_dim*w/volold[i]
 initialized_energy=np.sum(e*volold)
 lp=np.zeros(nt-1);ld=np.zeros(nt-1);qp=np.zeros(nt-1);kap=np.zeros(nt-1);budget_max=0.;growth=0;nonlinear_failed=0;max_iterations=0
 lower=np.zeros(nx);upper=np.zeros(nx);diag=np.ones(nx);rhs=np.zeros(nx);cp=np.zeros(nx);dp=np.zeros(nx);en=np.zeros(nx)
 ni_total=xi*moment(qmin,qni,2,qc,amp,n)
 for it in range(1,nt):
  f=1+t[it]*vmax/R0;dy=(t[it]-t[it-1])/tdiff
  converged=False
  klo=.34*floor_ratio;khi=.34;trial=min(max(kr[-1]*kref,klo),khi);last_k=trial;last_f=0.;have_last=False
  for iteration in range(28 if kind==2 else 1):
   if kind==2:
    kmid=trial
    for i in range(nx):kr[i]=kmid/kref if cent[i]>=qc else 1.
   qph=find_photo(faces,kr,(2/3)*f*f/tauscale,qc,amp,n)
   vol,rho,c,src,count=geometry(faces,qph,qni,xi,qc,amp,n)
   lower[:]=0.;upper[:]=0.;diag[:]=1.;rhs[:]=0.
   swept=0.
   for i in range(nx):swept+=max(volold[i]-vol[i],0.)*max(e[i],0.)
   for i in range(count-1):
    transport=faces[i+1]**2*2/(rho[i]*kr[i]+rho[i+1]*kr[i+1])/(c[i+1]-c[i])
    if exact_resistance:transport=1/moment(c[i],c[i+1],-2,qc,amp,n)
    upper[i]=dy*transport/vol[i];lower[i+1]=dy*transport/vol[i+1]
   trans=0.;bc=0.
   if count>0:
    bd=eta(qph,qc,amp,n)*kr[count-1];distance=max(qph-c[count-1],0.)
    alpha=4*f*f/(3*tauscale*bd)
    trans=qph*qph/(bd*(distance+alpha))
    if exact_resistance:trans=1/(moment(c[count-1],qph,-2,qc,amp,n)+4*f*f/(3*tauscale*qph*qph))
    bc=dy*trans/vol[count-1]
   for i in range(count):
    diag[i]+=f*(lower[i]+upper[i]+(bc if i==count-1 else 0.))
    rhs[i]=e[i]*min(volold[i],vol[i])/vol[i]+dy*f*heat[it]*src[i]
   for i in range(nx):
    aa=-f*lower[i];cc=-f*upper[i]
    den=diag[i] if i==0 else diag[i]-aa*cp[i-1]
    cp[i]=cc/den;dp[i]=(rhs[i] if i==0 else rhs[i]-aa*dp[i-1])/den
   en[-1]=dp[-1]
   for i in range(nx-2,-1,-1):en[i]=dp[i]-cp[i]*en[i+1]
   flux=trans*en[count-1] if count>0 else 0.
   source=np.sum(src*vol)
   if kind!=2:
    converged=True
    break
   lum_guess=max(L0*(flux+swept/(f*dy)),0.)
   teff=(lum_guess/(4*np.pi*SIGMA_SB*(R0*f*qph)**2))**.25 if count>0 else 0.
   fraction=.5*(1+np.tanh((teff-Trec)/800.))
   target=.34*(floor_ratio+(1-floor_ratio)*fraction)
   if abs(kmid-target)<3.4e-7 or khi-klo<3.4e-7:
    converged=True
    break
   root_f=kmid-target
   if root_f>0:khi=kmid
   else:klo=kmid
   candidate=.5*(klo+khi)
   if have_last and abs(root_f-last_f)>1e-15:
    secant=kmid-root_f*(kmid-last_k)/(root_f-last_f)
    if klo<secant<khi:candidate=secant
   last_k=kmid;last_f=root_f;have_last=True;trial=candidate
  if not converged:nonlinear_failed+=1
  max_iterations=max(max_iterations,iteration+1)
  if qph>qold+1e-8:growth+=1
  error=np.sum(en*vol)-np.sum(e*volold)+swept+dy*f*flux-dy*f*heat[it]*source
  budget_max=max(budget_max,abs(error)/max(np.sum(e*volold)+dy*f*heat[it]*source,1e-99))
  luminosity=L0*(flux+swept/(f*dy))
  if count>0:lp[it-1]=luminosity
  elif it>1:lp[it-2]+=luminosity
  ld[it-1]=L0*heat[it]*max(ni_total-source,0.) # L0 = 4pi rho0 (eNi-eCo) R0^3
  qp[it-1]=qph;kap[it-1]=kref*(kr[count-1] if count>0 else kr[0])
  e[:]=en;volold=vol;qold=qph
 return lp,ld,qp,kap,budget_max,initialized_energy/Eth_dim,growth,nonlinear_failed,max_iterations


def solve(p,mode,nx=160,ny=2400):
 kind={'baseline':0,'structure':1,'structure_sed':1,'opacity':2,'thermal':3,'thermal_sed':4,'sed_only':0}[mode]
 qc=p.get('q_core',1/3) if kind==1 else 1/3
 n=p.get('n',2.) if kind==1 else 10.
 amp=1.
 if kind==1:
  env=p['f_env'];amp=env/(1-env)*integ(1e-4,qc,2)/(qc**n*integ(qc,1.,2-n))
 if p.get('continuous',False):amp=1.
 im=moment(1e-4,1.,2,qc,amp,n);ik=moment(1e-4,1.,4,qc,amp,n)
 mass=p['M_ej']*M_SUN;R0=p['R_0']*R_SUN;vel=p['v_ej']*1e9*np.sqrt(im/ik)
 rho=mass/(4*np.pi*R0**3*im);td=3*p['kappa']*rho*R0**2/C_LIGHT
 u0=rho*(EPSILON_NI-EPSILON_CO)*td;L0=4*np.pi*R0*C_LIGHT*u0/(3*p['kappa']*rho)
 left=1e-4;right=1.
 for _ in range(50):
  mid=(left+right)/2
  if moment(1e-4,mid,2,qc,amp,n)/im<p['f_ni']:left=mid
  else:right=mid
 qni=(left+right)/2;xi=(p['M_ni']/p['M_ej'])*im/moment(1e-4,qni,2,qc,amp,n)
 t=np.linspace(0,1,ny+1)**2*p.get("t_max_days",65)*DAY
 if p.get('event_grid',False):
  if kind!=1:raise ValueError('Event grid currently validated for constant-opacity structure only')
  # Analytic time at which tau=2/3 reaches the density interface.
  event=(np.sqrt(1.5*p['kappa']*rho*R0*moment(qc,1.,0,qc,amp,n))-1)*R0/vel
  obs=np.unique(np.asarray(p.get('t_eval_days',[]),float))*DAY
  # Resolve the output as the left time-step limit; h is a numerical tolerance, not a fit parameter.
  h=p.get('output_step_days',1e-5)*DAY
  extras=np.r_[obs,obs-h,obs-2*h,obs-4*h,event,event-h,event+h,event-4*h,event+4*h]
  extras=extras[(extras>0)&(extras<t[-1])]
  t=np.unique(np.r_[t,extras])
 heat=_radioactive_heating_shape(t,np.sqrt(3*p['kappa_gamma']*mass/(4*np.pi*(p['v_ej']*1e9)**2)))
 # Include the structural interface as an exact cell face.
 faces=np.r_[np.linspace(1e-4,qc,int(.6*nx)+1),np.linspace(qc,1.,nx-int(.6*nx)+1)[1:]] if kind==1 else (np.unique(np.r_[np.linspace(1e-4,1.,nx+1),qc]) if kind==2 else np.linspace(1e-4,1.,nx+1))
 lp,ld,qph,kap,budget,retained,growth,nonlinear_failed,max_iterations=solve_kernel(faces,t,heat,rho,R0,vel,p['kappa'],u0,td,L0,p['E_Th_in']*1e49/(4*np.pi*u0*R0**3),qni,xi,qc,amp,n,p.get('thermal_kind',kind),p.get('f_heat_env',.5),p.get('Trec_H',6000.),p.get('opacity_floor_ratio',.05),p.get('exact_resistance',False))
 rout=R0+vel*t[1:];valid=qph>1e-4+1e-10;rp=np.where(valid,rout*qph,np.nan);tp=(np.maximum(lp,0)/(4*np.pi*SIGMA_SB*rp**2))**.25
 state=NickelTransportState(t_s=t[1:],Lbol=lp+ld,Lphotospheric=lp,Ldirect=ld,q_ph=qph,Rph=rp,Tph=tp,Rhom=rout,photosphere_valid=valid,density_profile='broken_power_law')
 meta=dict(time_steps=len(t)-1,event_grid=p.get('event_grid',False),exact_resistance=p.get('exact_resistance',False),envelope_mass_fraction=float(moment(qc,1.,2,qc,amp,n)/im),envelope_kinetic_fraction=float(moment(qc,1.,4,qc,amp,n)/ik),outer_density_index=float(n),core_density_index=0.,opacity_closure=("H-envelope kappa(T_eff), implicit scalar root; core constant" if kind==2 else "constant gray opacity"),nonlinear_failed_steps=int(nonlinear_failed),nonlinear_max_iterations=int(max_iterations),discrete_energy_budget_max_relative_error=float(budget),initial_thermal_fraction_in_photosphere=float(retained),photosphere_outward_steps=int(growth),vmax_cm_s=float(vel),kappa_ph_min=float(kap.min()),kappa_ph_max=float(kap.max()),mass_integral=float(im),kinetic_integral=float(ik),q_core=float(qc),outer_density_amplitude=float(amp))
 return state,meta


# BM12 photon response curves and CALSPEC alpha_lyr_stis_005 Vega zero points.
EMBEDDED_PASSBANDS = {'B': {'wavelength_angstrom': [3600.0, 3605.0, 3610.0, 3615.0, 3620.0, 3625.0, 3630.0, 3635.0, 3640.0, 3645.0,
                               3650.0, 3655.0, 3660.0, 3665.0, 3670.0, 3675.0, 3680.0, 3685.0, 3690.0, 3695.0,
                               3700.0, 3705.0, 3710.0, 3715.0, 3720.0, 3725.0, 3730.0, 3735.0, 3740.0, 3745.0,
                               3750.0, 3755.0, 3760.0, 3765.0, 3770.0, 3775.0, 3780.0, 3785.0, 3790.0, 3795.0,
                               3800.0, 3805.0, 3810.0, 3815.0, 3820.0, 3825.0, 3830.0, 3835.0, 3840.0, 3845.0,
                               3850.0, 3855.0, 3860.0, 3865.0, 3870.0, 3875.0, 3880.0, 3885.0, 3890.0, 3895.0,
                               3900.0, 3905.0, 3910.0, 3915.0, 3920.0, 3925.0, 3930.0, 3935.0, 3940.0, 3945.0,
                               3950.0, 3955.0, 3960.0, 3965.0, 3970.0, 3975.0, 3980.0, 3985.0, 3990.0, 3995.0,
                               4000.0, 4005.0, 4010.0, 4015.0, 4020.0, 4025.0, 4030.0, 4035.0, 4040.0, 4045.0,
                               4050.0, 4055.0, 4060.0, 4065.0, 4070.0, 4075.0, 4080.0, 4085.0, 4090.0, 4095.0,
                               4100.0, 4105.0, 4110.0, 4115.0, 4120.0, 4125.0, 4130.0, 4135.0, 4140.0, 4145.0,
                               4150.0, 4155.0, 4160.0, 4165.0, 4170.0, 4175.0, 4180.0, 4185.0, 4190.0, 4195.0,
                               4200.0, 4205.0, 4210.0, 4215.0, 4220.0, 4225.0, 4230.0, 4235.0, 4240.0, 4245.0,
                               4250.0, 4255.0, 4260.0, 4265.0, 4270.0, 4275.0, 4280.0, 4285.0, 4290.0, 4295.0,
                               4300.0, 4305.0, 4310.0, 4315.0, 4320.0, 4325.0, 4330.0, 4335.0, 4340.0, 4345.0,
                               4350.0, 4355.0, 4360.0, 4365.0, 4370.0, 4375.0, 4380.0, 4385.0, 4390.0, 4395.0,
                               4400.0, 4405.0, 4410.0, 4415.0, 4420.0, 4425.0, 4430.0, 4435.0, 4440.0, 4445.0,
                               4450.0, 4455.0, 4460.0, 4465.0, 4470.0, 4475.0, 4480.0, 4485.0, 4490.0, 4495.0,
                               4500.0, 4505.0, 4510.0, 4515.0, 4520.0, 4525.0, 4530.0, 4535.0, 4540.0, 4545.0,
                               4550.0, 4555.0, 4560.0, 4565.0, 4570.0, 4575.0, 4580.0, 4585.0, 4590.0, 4595.0,
                               4600.0, 4605.0, 4610.0, 4615.0, 4620.0, 4625.0, 4630.0, 4635.0, 4640.0, 4645.0,
                               4650.0, 4655.0, 4660.0, 4665.0, 4670.0, 4675.0, 4680.0, 4685.0, 4690.0, 4695.0,
                               4700.0, 4705.0, 4710.0, 4715.0, 4720.0, 4725.0, 4730.0, 4735.0, 4740.0, 4745.0,
                               4750.0, 4755.0, 4760.0, 4765.0, 4770.0, 4775.0, 4780.0, 4785.0, 4790.0, 4795.0,
                               4800.0, 4805.0, 4810.0, 4815.0, 4820.0, 4825.0, 4830.0, 4835.0, 4840.0, 4845.0,
                               4850.0, 4855.0, 4860.0, 4865.0, 4870.0, 4875.0, 4880.0, 4885.0, 4890.0, 4895.0,
                               4900.0, 4905.0, 4910.0, 4915.0, 4920.0, 4925.0, 4930.0, 4935.0, 4940.0, 4945.0,
                               4950.0, 4955.0, 4960.0, 4965.0, 4970.0, 4975.0, 4980.0, 4985.0, 4990.0, 4995.0,
                               5000.0, 5005.0, 5010.0, 5015.0, 5020.0, 5025.0, 5030.0, 5035.0, 5040.0, 5045.0,
                               5050.0, 5055.0, 5060.0, 5065.0, 5070.0, 5075.0, 5080.0, 5085.0, 5090.0, 5095.0,
                               5100.0, 5105.0, 5110.0, 5115.0, 5120.0, 5125.0, 5130.0, 5135.0, 5140.0, 5145.0,
                               5150.0, 5155.0, 5160.0, 5165.0, 5170.0, 5175.0, 5180.0, 5185.0, 5190.0, 5195.0,
                               5200.0, 5205.0, 5210.0, 5215.0, 5220.0, 5225.0, 5230.0, 5235.0, 5240.0, 5245.0,
                               5250.0, 5255.0, 5260.0, 5265.0, 5270.0, 5275.0, 5280.0, 5285.0, 5290.0, 5295.0,
                               5300.0, 5305.0, 5310.0, 5315.0, 5320.0, 5325.0, 5330.0, 5335.0, 5340.0, 5345.0,
                               5350.0, 5355.0, 5360.0, 5365.0, 5370.0, 5375.0, 5380.0, 5385.0, 5390.0, 5395.0,
                               5400.0, 5405.0, 5410.0, 5415.0, 5420.0, 5425.0, 5430.0, 5435.0, 5440.0, 5445.0,
                               5450.0, 5455.0, 5460.0, 5465.0, 5470.0, 5475.0, 5480.0, 5485.0, 5490.0, 5495.0,
                               5500.0, 5505.0, 5510.0, 5515.0, 5520.0, 5525.0, 5530.0, 5535.0, 5540.0, 5545.0,
                               5550.0, 5555.0, 5560.0, 5565.0, 5570.0, 5575.0, 5580.0, 5585.0, 5590.0, 5595.0,
                               5600.0],
       'photon_response': [0.0, 0.00155, 0.0031, 0.00465, 0.0062, 0.00775, 0.0093, 0.01085, 0.0124, 0.01395,
                           0.0155, 0.01705, 0.0186, 0.02015, 0.0217, 0.02325, 0.0248, 0.02635, 0.0279,
                           0.02945, 0.031, 0.0363, 0.0416, 0.046900000000000004, 0.0522, 0.0575,
                           0.06280000000000001, 0.06810000000000001, 0.0734, 0.0787, 0.084, 0.0893,
                           0.09460000000000002, 0.09990000000000002, 0.10520000000000002, 0.11050000000000001,
                           0.11580000000000001, 0.12110000000000001, 0.1264, 0.1317, 0.137,
                           0.15935000000000002, 0.1817, 0.20405, 0.2264, 0.24875, 0.2711, 0.29345,
                           0.31579999999999997, 0.33814999999999995, 0.3605, 0.38284999999999997,
                           0.40519999999999995, 0.42754999999999993, 0.44989999999999997, 0.47224999999999995,
                           0.49459999999999993, 0.5169499999999999, 0.5392999999999999, 0.56165, 0.584,
                           0.60215, 0.6203, 0.63845, 0.6566, 0.67475, 0.6929, 0.71105, 0.7292, 0.74735,
                           0.7655, 0.78365, 0.8018, 0.81995, 0.8381, 0.85625, 0.8744, 0.89255, 0.9107,
                           0.92885, 0.947, 0.94965, 0.9522999999999999, 0.95495, 0.9576, 0.9602499999999999,
                           0.9629, 0.96555, 0.9682, 0.97085, 0.9735, 0.97615, 0.9788, 0.9814499999999999,
                           0.9841, 0.98675, 0.9894, 0.99205, 0.9947, 0.99735, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99785,
                           0.9957, 0.99355, 0.9914, 0.98925, 0.9871, 0.98495, 0.9828, 0.98065, 0.9785,
                           0.9763499999999999, 0.9742, 0.97205, 0.9699, 0.96775, 0.9656, 0.9634499999999999,
                           0.9612999999999999, 0.95915, 0.957, 0.9539, 0.9508, 0.9477, 0.9446, 0.9415, 0.9384,
                           0.9353, 0.9322, 0.9291, 0.9259999999999999, 0.9228999999999999, 0.9198, 0.9167,
                           0.9136, 0.9105, 0.9074, 0.9043, 0.9012, 0.8981, 0.895, 0.89035, 0.8857, 0.88105,
                           0.8764000000000001, 0.87175, 0.8671, 0.86245, 0.8578, 0.8531500000000001, 0.8485,
                           0.84385, 0.8392000000000001, 0.83455, 0.8299000000000001, 0.82525, 0.8206,
                           0.8159500000000001, 0.8113, 0.8066500000000001, 0.802, 0.796, 0.79, 0.784, 0.778,
                           0.772, 0.766, 0.76, 0.754, 0.748, 0.7420000000000001, 0.7360000000000001,
                           0.7300000000000001, 0.7240000000000001, 0.7180000000000001, 0.7120000000000001,
                           0.7060000000000001, 0.7000000000000001, 0.6940000000000001, 0.6880000000000001,
                           0.682, 0.6767500000000001, 0.6715, 0.66625, 0.661, 0.65575, 0.6505, 0.64525, 0.64,
                           0.63475, 0.6295, 0.62425, 0.619, 0.61375, 0.6084999999999999, 0.60325, 0.598,
                           0.59275, 0.5874999999999999, 0.5822499999999999, 0.577, 0.57185, 0.5667, 0.56155,
                           0.5564, 0.5512499999999999, 0.5460999999999999, 0.5409499999999999,
                           0.5357999999999999, 0.53065, 0.5255, 0.52035, 0.5152, 0.51005, 0.5049,
                           0.49974999999999997, 0.4946, 0.48944999999999994, 0.48429999999999995,
                           0.47914999999999996, 0.474, 0.46875, 0.46349999999999997, 0.45825,
                           0.45299999999999996, 0.44775, 0.4425, 0.43724999999999997, 0.432,
                           0.42674999999999996, 0.4215, 0.41625, 0.411, 0.40575, 0.4005, 0.39525, 0.39,
                           0.38475, 0.3795, 0.37425, 0.369, 0.36445, 0.3599, 0.35535, 0.3508, 0.34625, 0.3417,
                           0.33715, 0.3326, 0.32805, 0.3235, 0.31895, 0.3144, 0.30985, 0.3053, 0.30075,
                           0.2962, 0.29165, 0.2871, 0.28255, 0.278, 0.274, 0.27, 0.266, 0.262, 0.258, 0.254,
                           0.25, 0.24600000000000002, 0.24200000000000002, 0.23800000000000002, 0.234, 0.23,
                           0.226, 0.222, 0.21800000000000003, 0.21400000000000002, 0.21000000000000002,
                           0.20600000000000002, 0.202, 0.198, 0.19435000000000002, 0.1907,
                           0.18705000000000002, 0.1834, 0.17975000000000002, 0.1761, 0.17245000000000002,
                           0.1688, 0.16515000000000002, 0.1615, 0.15785000000000002, 0.1542,
                           0.15055000000000002, 0.1469, 0.14325000000000002, 0.1396, 0.13595000000000002,
                           0.1323, 0.12865000000000001, 0.125, 0.12265, 0.1203, 0.11795, 0.1156, 0.11325,
                           0.1109, 0.10855, 0.1062, 0.10385, 0.1015, 0.09915, 0.0968, 0.09445, 0.0921,
                           0.08975, 0.0874, 0.08505, 0.0827, 0.08035, 0.078, 0.0759, 0.0738, 0.0717, 0.0696,
                           0.0675, 0.0654, 0.0633, 0.0612, 0.0591, 0.057, 0.0549, 0.0528, 0.0507, 0.0486,
                           0.0465, 0.044399999999999995, 0.0423, 0.0402, 0.038099999999999995, 0.036, 0.0346,
                           0.0332, 0.031799999999999995, 0.030399999999999996, 0.028999999999999998, 0.0276,
                           0.026199999999999998, 0.0248, 0.023399999999999997, 0.022, 0.0206, 0.0192, 0.0178,
                           0.016399999999999998, 0.015, 0.0136, 0.012199999999999999, 0.010799999999999999,
                           0.0094, 0.008, 0.0076, 0.0072, 0.0068000000000000005, 0.0064, 0.006, 0.0056,
                           0.0052, 0.0048, 0.0044, 0.004, 0.0036, 0.0031999999999999997,
                           0.0027999999999999995, 0.0024, 0.0019999999999999996, 0.0015999999999999996,
                           0.0011999999999999997, 0.0007999999999999996, 0.00039999999999999953, 0.0],
       'zero_point_jy': 4089.8510577867337},
 'V': {'wavelength_angstrom': [4700.0, 4705.0, 4710.0, 4715.0, 4720.0, 4725.0, 4730.0, 4735.0, 4740.0, 4745.0,
                               4750.0, 4755.0, 4760.0, 4765.0, 4770.0, 4775.0, 4780.0, 4785.0, 4790.0, 4795.0,
                               4800.0, 4805.0, 4810.0, 4815.0, 4820.0, 4825.0, 4830.0, 4835.0, 4840.0, 4845.0,
                               4850.0, 4855.0, 4860.0, 4865.0, 4870.0, 4875.0, 4880.0, 4885.0, 4890.0, 4895.0,
                               4900.0, 4905.0, 4910.0, 4915.0, 4920.0, 4925.0, 4930.0, 4935.0, 4940.0, 4945.0,
                               4950.0, 4955.0, 4960.0, 4965.0, 4970.0, 4975.0, 4980.0, 4985.0, 4990.0, 4995.0,
                               5000.0, 5005.0, 5010.0, 5015.0, 5020.0, 5025.0, 5030.0, 5035.0, 5040.0, 5045.0,
                               5050.0, 5055.0, 5060.0, 5065.0, 5070.0, 5075.0, 5080.0, 5085.0, 5090.0, 5095.0,
                               5100.0, 5105.0, 5110.0, 5115.0, 5120.0, 5125.0, 5130.0, 5135.0, 5140.0, 5145.0,
                               5150.0, 5155.0, 5160.0, 5165.0, 5170.0, 5175.0, 5180.0, 5185.0, 5190.0, 5195.0,
                               5200.0, 5205.0, 5210.0, 5215.0, 5220.0, 5225.0, 5230.0, 5235.0, 5240.0, 5245.0,
                               5250.0, 5255.0, 5260.0, 5265.0, 5270.0, 5275.0, 5280.0, 5285.0, 5290.0, 5295.0,
                               5300.0, 5305.0, 5310.0, 5315.0, 5320.0, 5325.0, 5330.0, 5335.0, 5340.0, 5345.0,
                               5350.0, 5355.0, 5360.0, 5365.0, 5370.0, 5375.0, 5380.0, 5385.0, 5390.0, 5395.0,
                               5400.0, 5405.0, 5410.0, 5415.0, 5420.0, 5425.0, 5430.0, 5435.0, 5440.0, 5445.0,
                               5450.0, 5455.0, 5460.0, 5465.0, 5470.0, 5475.0, 5480.0, 5485.0, 5490.0, 5495.0,
                               5500.0, 5505.0, 5510.0, 5515.0, 5520.0, 5525.0, 5530.0, 5535.0, 5540.0, 5545.0,
                               5550.0, 5555.0, 5560.0, 5565.0, 5570.0, 5575.0, 5580.0, 5585.0, 5590.0, 5595.0,
                               5600.0, 5605.0, 5610.0, 5615.0, 5620.0, 5625.0, 5630.0, 5635.0, 5640.0, 5645.0,
                               5650.0, 5655.0, 5660.0, 5665.0, 5670.0, 5675.0, 5680.0, 5685.0, 5690.0, 5695.0,
                               5700.0, 5705.0, 5710.0, 5715.0, 5720.0, 5725.0, 5730.0, 5735.0, 5740.0, 5745.0,
                               5750.0, 5755.0, 5760.0, 5765.0, 5770.0, 5775.0, 5780.0, 5785.0, 5790.0, 5795.0,
                               5800.0, 5805.0, 5810.0, 5815.0, 5820.0, 5825.0, 5830.0, 5835.0, 5840.0, 5845.0,
                               5850.0, 5855.0, 5860.0, 5865.0, 5870.0, 5875.0, 5880.0, 5885.0, 5890.0, 5895.0,
                               5900.0, 5905.0, 5910.0, 5915.0, 5920.0, 5925.0, 5930.0, 5935.0, 5940.0, 5945.0,
                               5950.0, 5955.0, 5960.0, 5965.0, 5970.0, 5975.0, 5980.0, 5985.0, 5990.0, 5995.0,
                               6000.0, 6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0, 6045.0,
                               6050.0, 6055.0, 6060.0, 6065.0, 6070.0, 6075.0, 6080.0, 6085.0, 6090.0, 6095.0,
                               6100.0, 6105.0, 6110.0, 6115.0, 6120.0, 6125.0, 6130.0, 6135.0, 6140.0, 6145.0,
                               6150.0, 6155.0, 6160.0, 6165.0, 6170.0, 6175.0, 6180.0, 6185.0, 6190.0, 6195.0,
                               6200.0, 6205.0, 6210.0, 6215.0, 6220.0, 6225.0, 6230.0, 6235.0, 6240.0, 6245.0,
                               6250.0, 6255.0, 6260.0, 6265.0, 6270.0, 6275.0, 6280.0, 6285.0, 6290.0, 6295.0,
                               6300.0, 6305.0, 6310.0, 6315.0, 6320.0, 6325.0, 6330.0, 6335.0, 6340.0, 6345.0,
                               6350.0, 6355.0, 6360.0, 6365.0, 6370.0, 6375.0, 6380.0, 6385.0, 6390.0, 6395.0,
                               6400.0, 6405.0, 6410.0, 6415.0, 6420.0, 6425.0, 6430.0, 6435.0, 6440.0, 6445.0,
                               6450.0, 6455.0, 6460.0, 6465.0, 6470.0, 6475.0, 6480.0, 6485.0, 6490.0, 6495.0,
                               6500.0, 6505.0, 6510.0, 6515.0, 6520.0, 6525.0, 6530.0, 6535.0, 6540.0, 6545.0,
                               6550.0, 6555.0, 6560.0, 6565.0, 6570.0, 6575.0, 6580.0, 6585.0, 6590.0, 6595.0,
                               6600.0, 6605.0, 6610.0, 6615.0, 6620.0, 6625.0, 6630.0, 6635.0, 6640.0, 6645.0,
                               6650.0, 6655.0, 6660.0, 6665.0, 6670.0, 6675.0, 6680.0, 6685.0, 6690.0, 6695.0,
                               6700.0, 6705.0, 6710.0, 6715.0, 6720.0, 6725.0, 6730.0, 6735.0, 6740.0, 6745.0,
                               6750.0, 6755.0, 6760.0, 6765.0, 6770.0, 6775.0, 6780.0, 6785.0, 6790.0, 6795.0,
                               6800.0, 6805.0, 6810.0, 6815.0, 6820.0, 6825.0, 6830.0, 6835.0, 6840.0, 6845.0,
                               6850.0, 6855.0, 6860.0, 6865.0, 6870.0, 6875.0, 6880.0, 6885.0, 6890.0, 6895.0,
                               6900.0, 6905.0, 6910.0, 6915.0, 6920.0, 6925.0, 6930.0, 6935.0, 6940.0, 6945.0,
                               6950.0, 6955.0, 6960.0, 6965.0, 6970.0, 6975.0, 6980.0, 6985.0, 6990.0, 6995.0,
                               7000.0, 7005.0, 7010.0, 7015.0, 7020.0, 7025.0, 7030.0, 7035.0, 7040.0, 7045.0,
                               7050.0, 7055.0, 7060.0, 7065.0, 7070.0, 7075.0, 7080.0, 7085.0, 7090.0, 7095.0,
                               7100.0, 7105.0, 7110.0, 7115.0, 7120.0, 7125.0, 7130.0, 7135.0, 7140.0, 7145.0,
                               7150.0, 7155.0, 7160.0, 7165.0, 7170.0, 7175.0, 7180.0, 7185.0, 7190.0, 7195.0,
                               7200.0, 7205.0, 7210.0, 7215.0, 7220.0, 7225.0, 7230.0, 7235.0, 7240.0, 7245.0,
                               7250.0, 7255.0, 7260.0, 7265.0, 7270.0, 7275.0, 7280.0, 7285.0, 7290.0, 7295.0,
                               7300.0, 7305.0, 7310.0, 7315.0, 7320.0, 7325.0, 7330.0, 7335.0, 7340.0, 7345.0,
                               7350.0, 7355.0, 7360.0, 7365.0, 7370.0, 7375.0, 7380.0, 7385.0, 7390.0, 7395.0,
                               7400.0],
       'photon_response': [0.0, 0.00165, 0.0033, 0.0049499999999999995, 0.0066, 0.00825, 0.009899999999999999,
                           0.01155, 0.0132, 0.01485, 0.0165, 0.01815, 0.019799999999999998, 0.02145, 0.0231,
                           0.02475, 0.0264, 0.02805, 0.0297, 0.03135, 0.033, 0.04015, 0.0473, 0.05445, 0.0616,
                           0.06874999999999999, 0.0759, 0.08305, 0.0902, 0.09734999999999999, 0.1045, 0.11165,
                           0.11879999999999999, 0.12594999999999998, 0.1331, 0.14024999999999999,
                           0.14739999999999998, 0.15455, 0.16169999999999998, 0.16885, 0.176,
                           0.19144999999999998, 0.2069, 0.22235, 0.23779999999999998, 0.25325, 0.2687,
                           0.28414999999999996, 0.2996, 0.31505, 0.33049999999999996, 0.34595, 0.3614,
                           0.37684999999999996, 0.3923, 0.40775, 0.42319999999999997, 0.43865, 0.4541,
                           0.46954999999999997, 0.485, 0.5013, 0.5176, 0.5339, 0.5502, 0.5665, 0.5828, 0.5991,
                           0.6154000000000001, 0.6317, 0.648, 0.6643, 0.6806, 0.6969000000000001,
                           0.7132000000000001, 0.7295, 0.7458, 0.7621, 0.7784000000000001, 0.7947000000000001,
                           0.811, 0.8197500000000001, 0.8285, 0.83725, 0.8460000000000001, 0.85475, 0.8635,
                           0.8722500000000001, 0.881, 0.88975, 0.8985000000000001, 0.90725, 0.916, 0.92475,
                           0.9335, 0.94225, 0.951, 0.95975, 0.9685, 0.97725, 0.986, 0.9867,
                           0.9873999999999999, 0.9881, 0.9888, 0.9894999999999999, 0.9902, 0.9909, 0.9916,
                           0.9923, 0.993, 0.9937, 0.9944, 0.9951, 0.9958, 0.9964999999999999, 0.9972, 0.9979,
                           0.9986, 0.9993, 1.0, 0.99775, 0.9954999999999999, 0.99325, 0.991, 0.98875,
                           0.9864999999999999, 0.98425, 0.982, 0.97975, 0.9774999999999999, 0.97525, 0.973,
                           0.97075, 0.9684999999999999, 0.9662499999999999, 0.964, 0.96175,
                           0.9594999999999999, 0.9572499999999999, 0.955, 0.9505, 0.946, 0.9415,
                           0.9369999999999999, 0.9325, 0.9279999999999999, 0.9235, 0.9189999999999999, 0.9145,
                           0.91, 0.9055, 0.901, 0.8965, 0.892, 0.8875, 0.883, 0.8785, 0.874,
                           0.8694999999999999, 0.865, 0.85925, 0.8535, 0.84775, 0.842, 0.8362499999999999,
                           0.8305, 0.82475, 0.819, 0.81325, 0.8075, 0.80175, 0.796, 0.79025, 0.7845,
                           0.7787499999999999, 0.773, 0.76725, 0.7615, 0.75575, 0.75, 0.7453, 0.7406, 0.7359,
                           0.7312, 0.7265, 0.7218, 0.7171, 0.7124, 0.7077, 0.703, 0.6983, 0.6936,
                           0.6889000000000001, 0.6842, 0.6795, 0.6748000000000001, 0.6701, 0.6654,
                           0.6607000000000001, 0.656, 0.65045, 0.6449, 0.6393500000000001, 0.6338, 0.62825,
                           0.6227, 0.6171500000000001, 0.6116, 0.60605, 0.6005, 0.5949500000000001, 0.5894,
                           0.58385, 0.5783, 0.5727500000000001, 0.5672, 0.56165, 0.5561, 0.5505500000000001,
                           0.545, 0.53945, 0.5339, 0.52835, 0.5228, 0.51725, 0.5117, 0.50615, 0.5006, 0.49505,
                           0.4895, 0.48395, 0.4784, 0.47285, 0.4673, 0.46175, 0.4562, 0.45065, 0.4451,
                           0.43955, 0.434, 0.429, 0.424, 0.419, 0.414, 0.409, 0.404, 0.399, 0.394, 0.389,
                           0.384, 0.379, 0.374, 0.369, 0.364, 0.359, 0.35400000000000004, 0.34900000000000003,
                           0.34400000000000003, 0.339, 0.334, 0.32975000000000004, 0.3255,
                           0.32125000000000004, 0.317, 0.31275000000000003, 0.3085, 0.30425, 0.3, 0.29575,
                           0.29150000000000004, 0.28725, 0.28300000000000003, 0.27875, 0.2745, 0.27025, 0.266,
                           0.26175, 0.2575, 0.25325000000000003, 0.249, 0.24555, 0.2421, 0.23865, 0.2352,
                           0.23174999999999998, 0.2283, 0.22485, 0.22139999999999999, 0.21795, 0.2145,
                           0.21105, 0.2076, 0.20415, 0.2007, 0.19724999999999998, 0.1938, 0.19035,
                           0.18689999999999998, 0.18345, 0.18, 0.1772, 0.1744, 0.1716, 0.1688,
                           0.16599999999999998, 0.16319999999999998, 0.1604, 0.1576, 0.1548, 0.152, 0.1492,
                           0.1464, 0.1436, 0.1408, 0.13799999999999998, 0.1352, 0.1324, 0.1296, 0.1268, 0.124,
                           0.12155, 0.1191, 0.11665, 0.1142, 0.11175, 0.1093, 0.10685, 0.10439999999999999,
                           0.10195, 0.0995, 0.09705, 0.0946, 0.09215, 0.0897, 0.08725, 0.0848, 0.08235,
                           0.0799, 0.07745, 0.075, 0.0733, 0.0716, 0.0699, 0.0682, 0.0665, 0.0648, 0.0631,
                           0.061399999999999996, 0.059699999999999996, 0.057999999999999996,
                           0.056299999999999996, 0.054599999999999996, 0.0529, 0.0512, 0.0495, 0.0478, 0.0461,
                           0.0444, 0.0427, 0.041, 0.04005, 0.0391, 0.03815, 0.037200000000000004, 0.03625,
                           0.0353, 0.03435, 0.0334, 0.03245, 0.0315, 0.03055, 0.029599999999999998, 0.02865,
                           0.0277, 0.02675, 0.0258, 0.024849999999999997, 0.023899999999999998,
                           0.022949999999999998, 0.022, 0.021599999999999998, 0.0212, 0.0208,
                           0.020399999999999998, 0.02, 0.0196, 0.0192, 0.0188, 0.0184, 0.018, 0.0176, 0.0172,
                           0.0168, 0.0164, 0.016, 0.015600000000000001, 0.0152, 0.0148, 0.014400000000000001,
                           0.014, 0.01385, 0.0137, 0.01355, 0.0134, 0.01325, 0.0131, 0.01295, 0.0128, 0.01265,
                           0.012499999999999999, 0.01235, 0.012199999999999999, 0.01205, 0.011899999999999999,
                           0.01175, 0.0116, 0.01145, 0.0113, 0.011149999999999998, 0.011, 0.01085, 0.0107,
                           0.01055, 0.0104, 0.01025, 0.0101, 0.00995, 0.0098, 0.00965, 0.0095, 0.00935,
                           0.0092, 0.00905, 0.0089, 0.00875, 0.0086, 0.008450000000000001, 0.0083,
                           0.008150000000000001, 0.008, 0.0079, 0.0078000000000000005, 0.0077, 0.0076, 0.0075,
                           0.0074, 0.0073, 0.0072, 0.0071, 0.007, 0.0069, 0.0068000000000000005, 0.0067,
                           0.0066, 0.0065, 0.0064, 0.0063, 0.0062, 0.0061, 0.006, 0.0059,
                           0.0058000000000000005, 0.0057, 0.0056, 0.0055, 0.0054, 0.0053, 0.0052, 0.0051,
                           0.005, 0.0049, 0.0048000000000000004, 0.0047, 0.0046, 0.0045, 0.0044, 0.0043,
                           0.0042, 0.0041, 0.004, 0.0039000000000000003, 0.0038, 0.0037, 0.0036, 0.0035,
                           0.0034000000000000002, 0.0033, 0.0032, 0.0031, 0.003, 0.0029, 0.0028, 0.0027,
                           0.0026, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021, 0.002, 0.0019500000000000001,
                           0.0019, 0.00185, 0.0018, 0.00175, 0.0017000000000000001, 0.00165, 0.0016, 0.00155,
                           0.0015, 0.00145, 0.0014, 0.00135, 0.0013, 0.00125, 0.0012, 0.00115, 0.0011,
                           0.00105, 0.001, 0.00095, 0.0009, 0.0008500000000000001, 0.0008, 0.00075, 0.0007,
                           0.00065, 0.0006, 0.00055, 0.0005, 0.00045, 0.00039999999999999996,
                           0.00034999999999999994, 0.0003, 0.00024999999999999995, 0.00019999999999999996,
                           0.00014999999999999996, 9.999999999999995e-05, 4.999999999999994e-05, 0.0],
       'zero_point_jy': 3699.1367289599066},
 'R': {'wavelength_angstrom': [5500.0, 5505.0, 5510.0, 5515.0, 5520.0, 5525.0, 5530.0, 5535.0, 5540.0, 5545.0,
                               5550.0, 5555.0, 5560.0, 5565.0, 5570.0, 5575.0, 5580.0, 5585.0, 5590.0, 5595.0,
                               5600.0, 5605.0, 5610.0, 5615.0, 5620.0, 5625.0, 5630.0, 5635.0, 5640.0, 5645.0,
                               5650.0, 5655.0, 5660.0, 5665.0, 5670.0, 5675.0, 5680.0, 5685.0, 5690.0, 5695.0,
                               5700.0, 5705.0, 5710.0, 5715.0, 5720.0, 5725.0, 5730.0, 5735.0, 5740.0, 5745.0,
                               5750.0, 5755.0, 5760.0, 5765.0, 5770.0, 5775.0, 5780.0, 5785.0, 5790.0, 5795.0,
                               5800.0, 5805.0, 5810.0, 5815.0, 5820.0, 5825.0, 5830.0, 5835.0, 5840.0, 5845.0,
                               5850.0, 5855.0, 5860.0, 5865.0, 5870.0, 5875.0, 5880.0, 5885.0, 5890.0, 5895.0,
                               5900.0, 5905.0, 5910.0, 5915.0, 5920.0, 5925.0, 5930.0, 5935.0, 5940.0, 5945.0,
                               5950.0, 5955.0, 5960.0, 5965.0, 5970.0, 5975.0, 5980.0, 5985.0, 5990.0, 5995.0,
                               6000.0, 6005.0, 6010.0, 6015.0, 6020.0, 6025.0, 6030.0, 6035.0, 6040.0, 6045.0,
                               6050.0, 6055.0, 6060.0, 6065.0, 6070.0, 6075.0, 6080.0, 6085.0, 6090.0, 6095.0,
                               6100.0, 6105.0, 6110.0, 6115.0, 6120.0, 6125.0, 6130.0, 6135.0, 6140.0, 6145.0,
                               6150.0, 6155.0, 6160.0, 6165.0, 6170.0, 6175.0, 6180.0, 6185.0, 6190.0, 6195.0,
                               6200.0, 6205.0, 6210.0, 6215.0, 6220.0, 6225.0, 6230.0, 6235.0, 6240.0, 6245.0,
                               6250.0, 6255.0, 6260.0, 6265.0, 6270.0, 6275.0, 6280.0, 6285.0, 6290.0, 6295.0,
                               6300.0, 6305.0, 6310.0, 6315.0, 6320.0, 6325.0, 6330.0, 6335.0, 6340.0, 6345.0,
                               6350.0, 6355.0, 6360.0, 6365.0, 6370.0, 6375.0, 6380.0, 6385.0, 6390.0, 6395.0,
                               6400.0, 6405.0, 6410.0, 6415.0, 6420.0, 6425.0, 6430.0, 6435.0, 6440.0, 6445.0,
                               6450.0, 6455.0, 6460.0, 6465.0, 6470.0, 6475.0, 6480.0, 6485.0, 6490.0, 6495.0,
                               6500.0, 6505.0, 6510.0, 6515.0, 6520.0, 6525.0, 6530.0, 6535.0, 6540.0, 6545.0,
                               6550.0, 6555.0, 6560.0, 6565.0, 6570.0, 6575.0, 6580.0, 6585.0, 6590.0, 6595.0,
                               6600.0, 6605.0, 6610.0, 6615.0, 6620.0, 6625.0, 6630.0, 6635.0, 6640.0, 6645.0,
                               6650.0, 6655.0, 6660.0, 6665.0, 6670.0, 6675.0, 6680.0, 6685.0, 6690.0, 6695.0,
                               6700.0, 6705.0, 6710.0, 6715.0, 6720.0, 6725.0, 6730.0, 6735.0, 6740.0, 6745.0,
                               6750.0, 6755.0, 6760.0, 6765.0, 6770.0, 6775.0, 6780.0, 6785.0, 6790.0, 6795.0,
                               6800.0, 6805.0, 6810.0, 6815.0, 6820.0, 6825.0, 6830.0, 6835.0, 6840.0, 6845.0,
                               6850.0, 6855.0, 6860.0, 6865.0, 6870.0, 6875.0, 6880.0, 6885.0, 6890.0, 6895.0,
                               6900.0, 6905.0, 6910.0, 6915.0, 6920.0, 6925.0, 6930.0, 6935.0, 6940.0, 6945.0,
                               6950.0, 6955.0, 6960.0, 6965.0, 6970.0, 6975.0, 6980.0, 6985.0, 6990.0, 6995.0,
                               7000.0, 7005.0, 7010.0, 7015.0, 7020.0, 7025.0, 7030.0, 7035.0, 7040.0, 7045.0,
                               7050.0, 7055.0, 7060.0, 7065.0, 7070.0, 7075.0, 7080.0, 7085.0, 7090.0, 7095.0,
                               7100.0, 7105.0, 7110.0, 7115.0, 7120.0, 7125.0, 7130.0, 7135.0, 7140.0, 7145.0,
                               7150.0, 7155.0, 7160.0, 7165.0, 7170.0, 7175.0, 7180.0, 7185.0, 7190.0, 7195.0,
                               7200.0, 7205.0, 7210.0, 7215.0, 7220.0, 7225.0, 7230.0, 7235.0, 7240.0, 7245.0,
                               7250.0, 7255.0, 7260.0, 7265.0, 7270.0, 7275.0, 7280.0, 7285.0, 7290.0, 7295.0,
                               7300.0, 7305.0, 7310.0, 7315.0, 7320.0, 7325.0, 7330.0, 7335.0, 7340.0, 7345.0,
                               7350.0, 7355.0, 7360.0, 7365.0, 7370.0, 7375.0, 7380.0, 7385.0, 7390.0, 7395.0,
                               7400.0, 7405.0, 7410.0, 7415.0, 7420.0, 7425.0, 7430.0, 7435.0, 7440.0, 7445.0,
                               7450.0, 7455.0, 7460.0, 7465.0, 7470.0, 7475.0, 7480.0, 7485.0, 7490.0, 7495.0,
                               7500.0, 7505.0, 7510.0, 7515.0, 7520.0, 7525.0, 7530.0, 7535.0, 7540.0, 7545.0,
                               7550.0, 7555.0, 7560.0, 7565.0, 7570.0, 7575.0, 7580.0, 7585.0, 7590.0, 7595.0,
                               7600.0, 7605.0, 7610.0, 7615.0, 7620.0, 7625.0, 7630.0, 7635.0, 7640.0, 7645.0,
                               7650.0, 7655.0, 7660.0, 7665.0, 7670.0, 7675.0, 7680.0, 7685.0, 7690.0, 7695.0,
                               7700.0, 7705.0, 7710.0, 7715.0, 7720.0, 7725.0, 7730.0, 7735.0, 7740.0, 7745.0,
                               7750.0, 7755.0, 7760.0, 7765.0, 7770.0, 7775.0, 7780.0, 7785.0, 7790.0, 7795.0,
                               7800.0, 7805.0, 7810.0, 7815.0, 7820.0, 7825.0, 7830.0, 7835.0, 7840.0, 7845.0,
                               7850.0, 7855.0, 7860.0, 7865.0, 7870.0, 7875.0, 7880.0, 7885.0, 7890.0, 7895.0,
                               7900.0, 7905.0, 7910.0, 7915.0, 7920.0, 7925.0, 7930.0, 7935.0, 7940.0, 7945.0,
                               7950.0, 7955.0, 7960.0, 7965.0, 7970.0, 7975.0, 7980.0, 7985.0, 7990.0, 7995.0,
                               8000.0, 8005.0, 8010.0, 8015.0, 8020.0, 8025.0, 8030.0, 8035.0, 8040.0, 8045.0,
                               8050.0, 8055.0, 8060.0, 8065.0, 8070.0, 8075.0, 8080.0, 8085.0, 8090.0, 8095.0,
                               8100.0, 8105.0, 8110.0, 8115.0, 8120.0, 8125.0, 8130.0, 8135.0, 8140.0, 8145.0,
                               8150.0, 8155.0, 8160.0, 8165.0, 8170.0, 8175.0, 8180.0, 8185.0, 8190.0, 8195.0,
                               8200.0, 8205.0, 8210.0, 8215.0, 8220.0, 8225.0, 8230.0, 8235.0, 8240.0, 8245.0,
                               8250.0, 8255.0, 8260.0, 8265.0, 8270.0, 8275.0, 8280.0, 8285.0, 8290.0, 8295.0,
                               8300.0, 8305.0, 8310.0, 8315.0, 8320.0, 8325.0, 8330.0, 8335.0, 8340.0, 8345.0,
                               8350.0, 8355.0, 8360.0, 8365.0, 8370.0, 8375.0, 8380.0, 8385.0, 8390.0, 8395.0,
                               8400.0, 8405.0, 8410.0, 8415.0, 8420.0, 8425.0, 8430.0, 8435.0, 8440.0, 8445.0,
                               8450.0, 8455.0, 8460.0, 8465.0, 8470.0, 8475.0, 8480.0, 8485.0, 8490.0, 8495.0,
                               8500.0, 8505.0, 8510.0, 8515.0, 8520.0, 8525.0, 8530.0, 8535.0, 8540.0, 8545.0,
                               8550.0, 8555.0, 8560.0, 8565.0, 8570.0, 8575.0, 8580.0, 8585.0, 8590.0, 8595.0,
                               8600.0, 8605.0, 8610.0, 8615.0, 8620.0, 8625.0, 8630.0, 8635.0, 8640.0, 8645.0,
                               8650.0, 8655.0, 8660.0, 8665.0, 8670.0, 8675.0, 8680.0, 8685.0, 8690.0, 8695.0,
                               8700.0, 8705.0, 8710.0, 8715.0, 8720.0, 8725.0, 8730.0, 8735.0, 8740.0, 8745.0,
                               8750.0, 8755.0, 8760.0, 8765.0, 8770.0, 8775.0, 8780.0, 8785.0, 8790.0, 8795.0,
                               8800.0, 8805.0, 8810.0, 8815.0, 8820.0, 8825.0, 8830.0, 8835.0, 8840.0, 8845.0,
                               8850.0, 8855.0, 8860.0, 8865.0, 8870.0, 8875.0, 8880.0, 8885.0, 8890.0, 8895.0,
                               8900.0, 8905.0, 8910.0, 8915.0, 8920.0, 8925.0, 8930.0, 8935.0, 8940.0, 8945.0,
                               8950.0, 8955.0, 8960.0, 8965.0, 8970.0, 8975.0, 8980.0, 8985.0, 8990.0, 8995.0,
                               9000.0, 9005.0, 9010.0, 9015.0, 9020.0, 9025.0, 9030.0, 9035.0, 9040.0, 9045.0,
                               9050.0, 9055.0, 9060.0, 9065.0, 9070.0, 9075.0, 9080.0, 9085.0, 9090.0, 9095.0,
                               9100.0],
       'photon_response': [0.0, 0.01235, 0.0247, 0.03705, 0.0494, 0.06175, 0.0741, 0.08645, 0.0988, 0.11115,
                           0.1235, 0.13585, 0.1482, 0.16055, 0.1729, 0.18525, 0.1976, 0.20995, 0.2223,
                           0.23465, 0.247, 0.27365, 0.3003, 0.32695, 0.3536, 0.38025000000000003,
                           0.40690000000000004, 0.43355, 0.4602, 0.48685, 0.5135000000000001, 0.54015,
                           0.5668000000000001, 0.59345, 0.6201, 0.64675, 0.6734, 0.7000500000000001, 0.7267,
                           0.7533500000000001, 0.78, 0.7881, 0.7962, 0.8043, 0.8124, 0.8205, 0.8286, 0.8367,
                           0.8448, 0.8529, 0.861, 0.8691, 0.8772, 0.8853, 0.8934, 0.9015, 0.9096, 0.9177,
                           0.9258, 0.9339, 0.942, 0.9448, 0.9476, 0.9503999999999999, 0.9531999999999999,
                           0.956, 0.9588, 0.9616, 0.9643999999999999, 0.9672, 0.97, 0.9728, 0.9756,
                           0.9783999999999999, 0.9812, 0.984, 0.9868, 0.9896, 0.9924, 0.9952, 0.998, 0.9981,
                           0.9982, 0.9983, 0.9984, 0.9985, 0.9986, 0.9987, 0.9988, 0.9989, 0.999, 0.9991,
                           0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999, 1.0, 0.9987,
                           0.9974, 0.9961, 0.9948, 0.9934999999999999, 0.9922, 0.9909, 0.9896, 0.9883, 0.987,
                           0.9857, 0.9843999999999999, 0.9831, 0.9818, 0.9804999999999999, 0.9792, 0.9779,
                           0.9766, 0.9753, 0.974, 0.9722999999999999, 0.9706, 0.9689, 0.9672, 0.9655, 0.9638,
                           0.9621, 0.9603999999999999, 0.9587, 0.957, 0.9552999999999999, 0.9536, 0.9519,
                           0.9501999999999999, 0.9485, 0.9468, 0.9450999999999999, 0.9433999999999999, 0.9417,
                           0.94, 0.9380499999999999, 0.9360999999999999, 0.9341499999999999,
                           0.9321999999999999, 0.93025, 0.9283, 0.92635, 0.9244, 0.92245, 0.9205, 0.91855,
                           0.9166, 0.91465, 0.9127, 0.9107500000000001, 0.9088, 0.90685, 0.9049, 0.90295,
                           0.901, 0.8989, 0.8968, 0.8947, 0.8926000000000001, 0.8905, 0.8884, 0.8863, 0.8842,
                           0.8821, 0.88, 0.8779, 0.8758, 0.8737, 0.8716, 0.8694999999999999, 0.8674, 0.8653,
                           0.8632, 0.8611, 0.859, 0.85675, 0.8544999999999999, 0.85225, 0.85, 0.84775,
                           0.8454999999999999, 0.8432499999999999, 0.841, 0.83875, 0.8364999999999999,
                           0.8342499999999999, 0.832, 0.82975, 0.8274999999999999, 0.8252499999999999, 0.823,
                           0.82075, 0.8184999999999999, 0.8162499999999999, 0.814, 0.8112999999999999, 0.8086,
                           0.8059, 0.8031999999999999, 0.8005, 0.7978, 0.7950999999999999, 0.7924, 0.7897,
                           0.787, 0.7843, 0.7816, 0.7789, 0.7762, 0.7735, 0.7708, 0.7681, 0.7654, 0.7627,
                           0.76, 0.75765, 0.7553, 0.75295, 0.7506, 0.74825, 0.7459, 0.74355, 0.7412, 0.73885,
                           0.7364999999999999, 0.73415, 0.7318, 0.7294499999999999, 0.7271, 0.72475,
                           0.7223999999999999, 0.72005, 0.7177, 0.7153499999999999, 0.713, 0.7104499999999999,
                           0.7079, 0.7053499999999999, 0.7028, 0.7002499999999999, 0.6977, 0.6951499999999999,
                           0.6926, 0.6900499999999999, 0.6875, 0.68495, 0.6824, 0.67985, 0.6773, 0.67475,
                           0.6722, 0.66965, 0.6671, 0.66455, 0.662, 0.65915, 0.6563, 0.65345,
                           0.6506000000000001, 0.64775, 0.6449, 0.64205, 0.6392, 0.63635, 0.6335, 0.63065,
                           0.6278, 0.62495, 0.6221, 0.61925, 0.6164, 0.61355, 0.6107, 0.60785, 0.605, 0.6023,
                           0.5996, 0.5969, 0.5942, 0.5915, 0.5888, 0.5861, 0.5834, 0.5807, 0.5780000000000001,
                           0.5753, 0.5726, 0.5699000000000001, 0.5672, 0.5645, 0.5618000000000001, 0.5591,
                           0.5564, 0.5537000000000001, 0.551, 0.5483, 0.5456000000000001, 0.5429, 0.5402,
                           0.5375000000000001, 0.5348, 0.5321, 0.5294, 0.5267000000000001, 0.524, 0.5213,
                           0.5186000000000001, 0.5159, 0.5132, 0.5105000000000001, 0.5078, 0.5051, 0.5024,
                           0.4997, 0.497, 0.49445, 0.4919, 0.48935, 0.4868, 0.48425, 0.4817, 0.47915, 0.4766,
                           0.47405, 0.4715, 0.46895, 0.4664, 0.46385, 0.4613, 0.45875, 0.4562, 0.45365,
                           0.4511, 0.44855, 0.446, 0.44365, 0.4413, 0.43895, 0.4366, 0.43425, 0.4319, 0.42955,
                           0.4272, 0.42485, 0.4225, 0.42015, 0.4178, 0.41545000000000004, 0.4131, 0.41075,
                           0.40840000000000004, 0.40605, 0.4037, 0.40135000000000004, 0.399, 0.39655, 0.3941,
                           0.39165, 0.3892, 0.38675000000000004, 0.38430000000000003, 0.38185, 0.3794,
                           0.37695, 0.3745, 0.37205, 0.3696, 0.36715, 0.36469999999999997, 0.36225, 0.3598,
                           0.35735, 0.3549, 0.35245, 0.35, 0.34754999999999997, 0.34509999999999996,
                           0.34264999999999995, 0.3402, 0.33775, 0.3353, 0.33285, 0.33039999999999997,
                           0.32794999999999996, 0.3255, 0.32305, 0.3206, 0.31815, 0.3157, 0.31325,
                           0.31079999999999997, 0.30835, 0.3059, 0.30345, 0.301, 0.2988, 0.2966, 0.2944,
                           0.2922, 0.29, 0.2878, 0.2856, 0.2834, 0.2812, 0.27899999999999997, 0.2768, 0.2746,
                           0.2724, 0.2702, 0.268, 0.2658, 0.2636, 0.2614, 0.2592, 0.257, 0.2549, 0.2528,
                           0.2507, 0.24860000000000002, 0.2465, 0.2444, 0.24230000000000002, 0.2402, 0.2381,
                           0.23600000000000002, 0.2339, 0.2318, 0.22970000000000002, 0.2276, 0.2255, 0.2234,
                           0.2213, 0.2192, 0.2171, 0.215, 0.21309999999999998, 0.2112, 0.20929999999999999,
                           0.2074, 0.2055, 0.2036, 0.2017, 0.1998, 0.1979, 0.19599999999999998, 0.1941,
                           0.19219999999999998, 0.1903, 0.18839999999999998, 0.1865, 0.1846, 0.1827, 0.1808,
                           0.17889999999999998, 0.177, 0.17534999999999998, 0.1737, 0.17204999999999998,
                           0.1704, 0.16874999999999998, 0.1671, 0.16544999999999999, 0.1638, 0.16215, 0.1605,
                           0.15885, 0.15719999999999998, 0.15555, 0.15389999999999998, 0.15225,
                           0.15059999999999998, 0.14895, 0.1473, 0.14565, 0.144, 0.14259999999999998, 0.1412,
                           0.13979999999999998, 0.1384, 0.13699999999999998, 0.1356, 0.13419999999999999,
                           0.1328, 0.1314, 0.13, 0.1286, 0.1272, 0.1258, 0.1244, 0.123, 0.1216, 0.1202,
                           0.1188, 0.1174, 0.116, 0.11465, 0.1133, 0.11195000000000001, 0.1106, 0.10925,
                           0.1079, 0.10655, 0.1052, 0.10385, 0.1025, 0.10115, 0.0998, 0.09845,
                           0.09709999999999999, 0.09575, 0.0944, 0.09305, 0.09169999999999999, 0.09035, 0.089,
                           0.08785, 0.0867, 0.08555, 0.0844, 0.08325, 0.08209999999999999, 0.08095, 0.0798,
                           0.07865, 0.0775, 0.07635, 0.0752, 0.07405, 0.0729, 0.07175000000000001, 0.0706,
                           0.06945, 0.0683, 0.06715, 0.066, 0.06525, 0.0645, 0.06375, 0.063, 0.06225, 0.0615,
                           0.06075, 0.06, 0.05925, 0.058499999999999996, 0.05775, 0.057, 0.05625, 0.0555,
                           0.05475, 0.054, 0.05325, 0.0525, 0.05175, 0.051, 0.0504, 0.0498, 0.0492, 0.0486,
                           0.048, 0.0474, 0.0468, 0.0462, 0.0456, 0.045, 0.0444, 0.0438, 0.0432, 0.0426,
                           0.042, 0.0414, 0.0408, 0.0402, 0.0396, 0.039, 0.03855, 0.0381, 0.03765, 0.0372,
                           0.03675, 0.0363, 0.03585, 0.0354, 0.03495, 0.0345, 0.03405, 0.0336, 0.03315,
                           0.0327, 0.03225, 0.0318, 0.03135, 0.0309, 0.030449999999999998, 0.03, 0.02955,
                           0.029099999999999997, 0.02865, 0.0282, 0.02775, 0.027299999999999998, 0.02685,
                           0.0264, 0.02595, 0.0255, 0.02505, 0.0246, 0.02415, 0.0237, 0.02325, 0.0228,
                           0.022350000000000002, 0.0219, 0.02145, 0.021, 0.02065, 0.020300000000000002,
                           0.019950000000000002, 0.019600000000000003, 0.01925, 0.0189, 0.01855, 0.0182,
                           0.01785, 0.0175, 0.017150000000000002, 0.016800000000000002, 0.01645, 0.0161,
                           0.01575, 0.0154, 0.015050000000000001, 0.014700000000000001, 0.01435, 0.014,
                           0.0137, 0.0134, 0.0131, 0.0128, 0.0125, 0.0122, 0.0119, 0.011600000000000001,
                           0.011300000000000001, 0.011, 0.0107, 0.0104, 0.0101, 0.0098, 0.0095, 0.0092,
                           0.0089, 0.0086, 0.0083, 0.008, 0.0079, 0.0078000000000000005, 0.0077, 0.0076,
                           0.0075, 0.0074, 0.0073, 0.0072, 0.0071, 0.007, 0.0069, 0.0068000000000000005,
                           0.0067, 0.0066, 0.0065, 0.0064, 0.0063, 0.0062, 0.0061, 0.006, 0.00585, 0.0057,
                           0.00555, 0.0054, 0.00525, 0.0051, 0.00495, 0.0048000000000000004,
                           0.0046500000000000005, 0.0045, 0.00435, 0.0042, 0.00405, 0.0039000000000000003,
                           0.00375, 0.0036, 0.00345, 0.0033, 0.00315, 0.003, 0.00285, 0.0027, 0.00255,
                           0.0024000000000000002, 0.00225, 0.0021, 0.0019500000000000001, 0.0018, 0.00165,
                           0.0015, 0.00135, 0.0012000000000000001, 0.00105, 0.0009, 0.00075,
                           0.0006000000000000001, 0.00045, 0.0003, 0.00015, 0.0],
       'zero_point_jy': 3124.993343251599},
 'I': {'wavelength_angstrom': [7000.0, 7005.0, 7010.0, 7015.0, 7020.0, 7025.0, 7030.0, 7035.0, 7040.0, 7045.0,
                               7050.0, 7055.0, 7060.0, 7065.0, 7070.0, 7075.0, 7080.0, 7085.0, 7090.0, 7095.0,
                               7100.0, 7105.0, 7110.0, 7115.0, 7120.0, 7125.0, 7130.0, 7135.0, 7140.0, 7145.0,
                               7150.0, 7155.0, 7160.0, 7165.0, 7170.0, 7175.0, 7180.0, 7185.0, 7190.0, 7195.0,
                               7200.0, 7205.0, 7210.0, 7215.0, 7220.0, 7225.0, 7230.0, 7235.0, 7240.0, 7245.0,
                               7250.0, 7255.0, 7260.0, 7265.0, 7270.0, 7275.0, 7280.0, 7285.0, 7290.0, 7295.0,
                               7300.0, 7305.0, 7310.0, 7315.0, 7320.0, 7325.0, 7330.0, 7335.0, 7340.0, 7345.0,
                               7350.0, 7355.0, 7360.0, 7365.0, 7370.0, 7375.0, 7380.0, 7385.0, 7390.0, 7395.0,
                               7400.0, 7405.0, 7410.0, 7415.0, 7420.0, 7425.0, 7430.0, 7435.0, 7440.0, 7445.0,
                               7450.0, 7455.0, 7460.0, 7465.0, 7470.0, 7475.0, 7480.0, 7485.0, 7490.0, 7495.0,
                               7500.0, 7505.0, 7510.0, 7515.0, 7520.0, 7525.0, 7530.0, 7535.0, 7540.0, 7545.0,
                               7550.0, 7555.0, 7560.0, 7565.0, 7570.0, 7575.0, 7580.0, 7585.0, 7590.0, 7595.0,
                               7600.0, 7605.0, 7610.0, 7615.0, 7620.0, 7625.0, 7630.0, 7635.0, 7640.0, 7645.0,
                               7650.0, 7655.0, 7660.0, 7665.0, 7670.0, 7675.0, 7680.0, 7685.0, 7690.0, 7695.0,
                               7700.0, 7705.0, 7710.0, 7715.0, 7720.0, 7725.0, 7730.0, 7735.0, 7740.0, 7745.0,
                               7750.0, 7755.0, 7760.0, 7765.0, 7770.0, 7775.0, 7780.0, 7785.0, 7790.0, 7795.0,
                               7800.0, 7805.0, 7810.0, 7815.0, 7820.0, 7825.0, 7830.0, 7835.0, 7840.0, 7845.0,
                               7850.0, 7855.0, 7860.0, 7865.0, 7870.0, 7875.0, 7880.0, 7885.0, 7890.0, 7895.0,
                               7900.0, 7905.0, 7910.0, 7915.0, 7920.0, 7925.0, 7930.0, 7935.0, 7940.0, 7945.0,
                               7950.0, 7955.0, 7960.0, 7965.0, 7970.0, 7975.0, 7980.0, 7985.0, 7990.0, 7995.0,
                               8000.0, 8005.0, 8010.0, 8015.0, 8020.0, 8025.0, 8030.0, 8035.0, 8040.0, 8045.0,
                               8050.0, 8055.0, 8060.0, 8065.0, 8070.0, 8075.0, 8080.0, 8085.0, 8090.0, 8095.0,
                               8100.0, 8105.0, 8110.0, 8115.0, 8120.0, 8125.0, 8130.0, 8135.0, 8140.0, 8145.0,
                               8150.0, 8155.0, 8160.0, 8165.0, 8170.0, 8175.0, 8180.0, 8185.0, 8190.0, 8195.0,
                               8200.0, 8205.0, 8210.0, 8215.0, 8220.0, 8225.0, 8230.0, 8235.0, 8240.0, 8245.0,
                               8250.0, 8255.0, 8260.0, 8265.0, 8270.0, 8275.0, 8280.0, 8285.0, 8290.0, 8295.0,
                               8300.0, 8305.0, 8310.0, 8315.0, 8320.0, 8325.0, 8330.0, 8335.0, 8340.0, 8345.0,
                               8350.0, 8355.0, 8360.0, 8365.0, 8370.0, 8375.0, 8380.0, 8385.0, 8390.0, 8395.0,
                               8400.0, 8405.0, 8410.0, 8415.0, 8420.0, 8425.0, 8430.0, 8435.0, 8440.0, 8445.0,
                               8450.0, 8455.0, 8460.0, 8465.0, 8470.0, 8475.0, 8480.0, 8485.0, 8490.0, 8495.0,
                               8500.0, 8505.0, 8510.0, 8515.0, 8520.0, 8525.0, 8530.0, 8535.0, 8540.0, 8545.0,
                               8550.0, 8555.0, 8560.0, 8565.0, 8570.0, 8575.0, 8580.0, 8585.0, 8590.0, 8595.0,
                               8600.0, 8605.0, 8610.0, 8615.0, 8620.0, 8625.0, 8630.0, 8635.0, 8640.0, 8645.0,
                               8650.0, 8655.0, 8660.0, 8665.0, 8670.0, 8675.0, 8680.0, 8685.0, 8690.0, 8695.0,
                               8700.0, 8705.0, 8710.0, 8715.0, 8720.0, 8725.0, 8730.0, 8735.0, 8740.0, 8745.0,
                               8750.0, 8755.0, 8760.0, 8765.0, 8770.0, 8775.0, 8780.0, 8785.0, 8790.0, 8795.0,
                               8800.0, 8805.0, 8810.0, 8815.0, 8820.0, 8825.0, 8830.0, 8835.0, 8840.0, 8845.0,
                               8850.0, 8855.0, 8860.0, 8865.0, 8870.0, 8875.0, 8880.0, 8885.0, 8890.0, 8895.0,
                               8900.0, 8905.0, 8910.0, 8915.0, 8920.0, 8925.0, 8930.0, 8935.0, 8940.0, 8945.0,
                               8950.0, 8955.0, 8960.0, 8965.0, 8970.0, 8975.0, 8980.0, 8985.0, 8990.0, 8995.0,
                               9000.0, 9005.0, 9010.0, 9015.0, 9020.0, 9025.0, 9030.0, 9035.0, 9040.0, 9045.0,
                               9050.0, 9055.0, 9060.0, 9065.0, 9070.0, 9075.0, 9080.0, 9085.0, 9090.0, 9095.0,
                               9100.0, 9105.0, 9110.0, 9115.0, 9120.0, 9125.0, 9130.0, 9135.0, 9140.0, 9145.0,
                               9150.0, 9155.0, 9160.0, 9165.0, 9170.0, 9175.0, 9180.0, 9185.0, 9190.0, 9195.0,
                               9200.0],
       'photon_response': [0.0, 0.0045, 0.009, 0.0135, 0.018, 0.0225, 0.027, 0.0315, 0.036, 0.0405, 0.045,
                           0.049499999999999995, 0.054, 0.058499999999999996, 0.063, 0.0675, 0.072, 0.0765,
                           0.081, 0.08549999999999999, 0.09, 0.1033, 0.1166, 0.1299, 0.1432, 0.1565, 0.1698,
                           0.18309999999999998, 0.1964, 0.2097, 0.223, 0.2363, 0.2496, 0.2629, 0.2762, 0.2895,
                           0.3028, 0.3161, 0.3294, 0.3427, 0.356, 0.3711, 0.3862, 0.4013, 0.4164, 0.4315,
                           0.4466, 0.4617, 0.4768, 0.4919, 0.507, 0.5221, 0.5372, 0.5523, 0.5674, 0.5825,
                           0.5976, 0.6127, 0.6278, 0.6429, 0.658, 0.66835, 0.6787000000000001, 0.68905,
                           0.6994, 0.70975, 0.7201000000000001, 0.73045, 0.7408, 0.75115, 0.7615000000000001,
                           0.77185, 0.7822, 0.79255, 0.8029000000000001, 0.81325, 0.8236, 0.83395, 0.8443,
                           0.85465, 0.865, 0.86975, 0.8744999999999999, 0.87925, 0.884, 0.88875, 0.8935,
                           0.89825, 0.903, 0.90775, 0.9125, 0.91725, 0.9219999999999999, 0.92675, 0.9315,
                           0.93625, 0.941, 0.94575, 0.9505, 0.9552499999999999, 0.96, 0.962, 0.964, 0.966,
                           0.968, 0.97, 0.972, 0.974, 0.976, 0.978, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99,
                           0.992, 0.994, 0.996, 0.998, 1.0, 0.9999, 0.9998, 0.9997, 0.9996,
                           0.9994999999999999, 0.9994, 0.9993, 0.9992, 0.9991, 0.999, 0.9989, 0.9988, 0.9987,
                           0.9986, 0.9984999999999999, 0.9984, 0.9983, 0.9982, 0.9981, 0.998, 0.99735, 0.9967,
                           0.99605, 0.9954, 0.99475, 0.9941, 0.9934499999999999, 0.9928, 0.99215,
                           0.9914999999999999, 0.99085, 0.9902, 0.98955, 0.9889, 0.98825, 0.9876, 0.98695,
                           0.9863, 0.98565, 0.985, 0.9843999999999999, 0.9838, 0.9832, 0.9826, 0.982,
                           0.9813999999999999, 0.9808, 0.9802, 0.9796, 0.979, 0.9783999999999999, 0.9778,
                           0.9772, 0.9766, 0.976, 0.9753999999999999, 0.9748, 0.9742, 0.9736, 0.973, 0.97285,
                           0.9727, 0.97255, 0.9723999999999999, 0.97225, 0.9721, 0.97195, 0.9718, 0.97165,
                           0.9714999999999999, 0.9713499999999999, 0.9712, 0.97105, 0.9709, 0.97075, 0.9706,
                           0.9704499999999999, 0.9702999999999999, 0.97015, 0.97, 0.9693999999999999, 0.9688,
                           0.9682, 0.9676, 0.967, 0.9663999999999999, 0.9658, 0.9652, 0.9646, 0.964,
                           0.9633999999999999, 0.9628, 0.9621999999999999, 0.9616, 0.961, 0.9603999999999999,
                           0.9598, 0.9591999999999999, 0.9586, 0.958, 0.9567, 0.9554, 0.9541, 0.9528, 0.9515,
                           0.9502, 0.9489, 0.9476, 0.9463, 0.9450000000000001, 0.9437, 0.9424, 0.9411,
                           0.9398000000000001, 0.9385, 0.9372, 0.9359000000000001, 0.9346000000000001, 0.9333,
                           0.932, 0.9306000000000001, 0.9292, 0.9278000000000001, 0.9264, 0.925,
                           0.9236000000000001, 0.9222, 0.9208000000000001, 0.9194, 0.918, 0.9166000000000001,
                           0.9152, 0.9138000000000001, 0.9124, 0.911, 0.9096000000000001, 0.9082, 0.9068,
                           0.9054, 0.904, 0.9018, 0.8996000000000001, 0.8974, 0.8952, 0.893, 0.8908,
                           0.8886000000000001, 0.8864, 0.8842, 0.882, 0.8798, 0.8776, 0.8754, 0.8732, 0.871,
                           0.8688, 0.8666, 0.8644, 0.8622, 0.86, 0.8575, 0.855, 0.8525, 0.85, 0.8475, 0.845,
                           0.8425, 0.84, 0.8375, 0.835, 0.8325, 0.8300000000000001, 0.8275,
                           0.8250000000000001, 0.8225, 0.8200000000000001, 0.8175, 0.8150000000000001, 0.8125,
                           0.81, 0.8062, 0.8024, 0.7986000000000001, 0.7948000000000001, 0.791, 0.7872,
                           0.7834, 0.7796000000000001, 0.7758, 0.772, 0.7682, 0.7644, 0.7606, 0.7568, 0.753,
                           0.7492, 0.7454, 0.7416, 0.7378, 0.734, 0.7268, 0.7196, 0.7124, 0.7051999999999999,
                           0.698, 0.6908, 0.6836, 0.6764, 0.6692, 0.662, 0.6547999999999999, 0.6476, 0.6404,
                           0.6332, 0.626, 0.6188, 0.6116, 0.6043999999999999, 0.5972, 0.59, 0.5801,
                           0.5701999999999999, 0.5603, 0.5504, 0.5405, 0.5306, 0.5206999999999999, 0.5108,
                           0.5009, 0.491, 0.48109999999999997, 0.4712, 0.4613, 0.4514, 0.4415, 0.4316, 0.4217,
                           0.4118, 0.40190000000000003, 0.392, 0.38255, 0.37310000000000004,
                           0.36365000000000003, 0.3542, 0.34475, 0.33530000000000004, 0.32585000000000003,
                           0.3164, 0.30695, 0.29750000000000004, 0.28805000000000003, 0.2786, 0.26915,
                           0.25970000000000004, 0.25025000000000003, 0.24080000000000001, 0.23135000000000003,
                           0.22190000000000001, 0.21245000000000003, 0.203, 0.19635000000000002, 0.1897,
                           0.18305000000000002, 0.1764, 0.16975, 0.16310000000000002, 0.15645,
                           0.14980000000000002, 0.14315, 0.1365, 0.12985000000000002, 0.12320000000000002,
                           0.11655000000000001, 0.10990000000000001, 0.10325000000000001, 0.09660000000000002,
                           0.08995000000000002, 0.08330000000000001, 0.07665000000000001, 0.07, 0.0669,
                           0.06380000000000001, 0.060700000000000004, 0.057600000000000005,
                           0.05450000000000001, 0.0514, 0.0483, 0.045200000000000004, 0.0421, 0.039, 0.0359,
                           0.0328, 0.0297, 0.0266, 0.0235, 0.020399999999999998, 0.017299999999999996,
                           0.014199999999999997, 0.011099999999999997, 0.008, 0.0076, 0.0072,
                           0.0068000000000000005, 0.0064, 0.006, 0.0056, 0.0052, 0.0048, 0.0044, 0.004,
                           0.0036, 0.0031999999999999997, 0.0027999999999999995, 0.0024,
                           0.0019999999999999996, 0.0015999999999999996, 0.0011999999999999997,
                           0.0007999999999999996, 0.00039999999999999953, 0.0],
       'zero_point_jy': 2504.8010949371997}}

# === Likelihood and priors ===
COORDINATES=['log10_Mej','log10_Ek_1e51','log10_Eth_1e49','log10_Mni','log10_R0',
             'log10_Ni_abundance','log10_Tfloor','log10_kappa_gamma','tshift_days',
             'log10_fenv','n']
PHYSICAL_NAMES=['M_ej_Msun','E_k_1e51_erg','E_th_1e49_erg','M_Ni_Msun','R0_Rsun',
                'f_Ni','T_floor_K','kappa_gamma_cm2_g','tshift_days','f_env','n']
BOUND_NAMES=PHYSICAL_NAMES.copy();BOUND_NAMES[5]='Ni_mass_fraction_in_mixed_region'
LOG_INDICES=[0,1,2,3,4,5,6,7,9]
DERIVED_NAMES=['M_env_Msun','v_ej_km_s','t_gamma_days','Eth_over_Ekin','Ni_abundance']

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def physical_samples(x):
    a=np.asarray(x,float).copy()
    a[...,LOG_INDICES]=10.**a[...,LOG_INDICES]
    a[...,5]=a[...,3]/(a[...,0]*a[...,5])
    return a

def derived_samples(x):
    a=physical_samples(x)
    mej,ek,eth,mni,fni,kg,env=a[...,0],a[...,1]*1e51,a[...,2],a[...,3],a[...,5],a[...,7],a[...,9]
    v=np.sqrt(2*ek/(mej*M_SUN))
    return np.stack([mej*env,v/1e5,np.sqrt(3*kg*mej*M_SUN/(4*np.pi*v**2))/DAY,
                     eth*1e49/ek,mni/(mej*fni)],axis=-1)

class Model:
    def __init__(self,config,nx=None,ny=None):
        self.base=ROOT
        self.config=copy.deepcopy(config)
        c=self.config
        if nx is not None:c['nx']=nx
        if ny is not None:c['ny']=ny
        if c['prior'] not in ['velocity_uniform','energy_uniform']:raise ValueError('Unknown energy prior')
        vlo,vhi=c['velocity_prior_bounds']
        if not 0<vlo<vhi:raise ValueError('Invalid velocity prior bounds')
        if c['prior']=='velocity_uniform':
            mlo,mhi=c['bounds']['M_ej_Msun']
            c['bounds']['E_k_1e51_erg']=[.5*mlo*M_SUN*(vlo*1e9)**2/1e51,
                                       .5*mhi*M_SUN*(vhi*1e9)**2/1e51]
        if c['time_format'] not in ['JD','MJD']:raise ValueError('time_format must be JD or MJD; reference_epoch must use the same system')
        if c['data_already_dereddened'] and c['ebv']!=0:raise ValueError('Dereddened data require ebv=0 to avoid double correction')
        if c['distance_mpc']<=0 or c['redshift']<=-1 or c['rv']<=0 or c['ebv']<0:raise ValueError('Invalid distance/redshift/extinction')
        if c['sigma_systematic_mag']<0 or c['kappa']<=0 or not 1e-4<c['q_core']<1:raise ValueError('Invalid error/opacity/core boundary')
        if c['nx']<20 or c['ny']<100 or c['output_step_days']<=0:raise ValueError('Invalid numerical grid/tolerance')
        self.bounds=np.array([c['bounds'][k] for k in BOUND_NAMES],float)
        if not np.all(np.isfinite(self.bounds)) or np.any(self.bounds[:,0]>=self.bounds[:,1]):raise ValueError('Invalid bounds')
        if np.any(self.bounds[LOG_INDICES]<=0):raise ValueError('Log coordinates require positive bounds')
        if c['bounds']['Ni_mass_fraction_in_mixed_region'][1]>1 or c['bounds']['f_env'][1]>=1:raise ValueError('Invalid mass fractions')
        if not 0<c['bounds']['f_Ni'][0]<c['bounds']['f_Ni'][1]<=1:raise ValueError('Invalid f_Ni bounds')
        self.bounds[LOG_INDICES]=np.log10(self.bounds[LOG_INDICES])
        self.data_path=(self.base/c['data_file']).resolve()
        raw=np.genfromtxt(self.data_path,delimiter=',',names=True,encoding='utf-8-sig')
        raw=np.atleast_1d(raw);names=raw.dtype.names or ()
        required=[c['time_column']]+[v for b in c['bands'] for v in [b,b+'_err']]
        if any(k not in names for k in required):raise ValueError('CSV must contain: '+', '.join(required))
        rows=[]
        for b in c['bands']:
            if b not in 'BVRI' or len(b)!=1:raise ValueError('This bundle supplies only B,V,R,I passbands')
            for r in raw:
                absolute=float(r[c['time_column']]);t=absolute-c['reference_epoch'];mag=float(r[b]);err=float(r[b+'_err'])
                if not np.isfinite([absolute,mag,err]).all() or err<=0:continue
                if c['zero_is_missing'] and mag==0:continue
                if c['phase_min_days']<t<=c['phase_max_days']:rows.append((t,b,mag,err,absolute))
        rows.sort()
        if not rows:raise ValueError('No valid points: check time units, reference_epoch and phase window')
        self.t=np.array([r[0] for r in rows]);self.band=np.array([r[1] for r in rows])
        self.y=np.array([r[2] for r in rows]);self.err=np.array([r[3] for r in rows])
        self.absolute_time=np.array([r[4] for r in rows]);self.sigma=np.hypot(self.err,c['sigma_systematic_mag'])
        if self.t.min()<=self.bounds[8,1]:raise ValueError('Explosion prior allows nonpositive ages; lower its upper bound')
        if (self.t.max()-self.bounds[8,0])/(1+c['redshift'])>=c['t_max_days']:raise ValueError('Increase t_max_days to cover the entire explosion-time prior')
        self.tg=np.geomspace(1000,1e6,10000);self.sed=BlackbodySED()
        self.cal=EMBEDDED_PASSBANDS;self.passbands={};self.table={}
        for b in set(self.band):
            w=np.array(self.cal[b]['wavelength_angstrom']);r=np.array(self.cal[b]['photon_response'])
            attenuation=10**(-.4*c['ebv']*c['rv']*extinction_axav(w/1e4,law='ccm89',rv=c['rv']))
            self.passbands[b]=(w,r,attenuation)
            self.table[b]=self.direct_magnitude(b,self.tg)
        self.initial=np.array(INITIAL_COORDINATES,float)
        if not self.valid(self.initial):raise ValueError('Initial coordinates lie outside the configured physical support')

    def direct_magnitude(self,b,temp):
        w,r,att=self.passbands[b];c=self.config
        f=self.sed.fnu(C_LIGHT/(w*1e-8),np.atleast_1d(temp),np.ones(np.size(temp)),DL_cm=c['distance_mpc']*1e6*PC,z=c['redshift'])
        flux=np.trapezoid(f*(r/w*att)[:,None],w,axis=0)/np.trapezoid(r/w,w)
        return -2.5*np.log10(flux/(self.cal[b]['zero_point_jy']*1e-23))

    def unit_magnitude(self,b,temp):
        temp=np.asarray(temp);out=np.interp(np.log(temp),np.log(self.tg),self.table[b])
        outside=(temp<self.tg[0])|(temp>self.tg[-1])
        if np.any(outside):out[outside]=self.direct_magnitude(b,temp[outside])
        return out

    def valid(self,x):
        x=np.asarray(x)
        if x.shape!=(11,) or not np.isfinite(x).all() or np.any(x<self.bounds[:,0]) or np.any(x>self.bounds[:,1]):return False
        if self.config['prior']=='velocity_uniform':
            vel=np.sqrt(2*10.**x[1]*1e51/(10.**x[0]*M_SUN))/1e9
            vlo,vhi=self.config['velocity_prior_bounds']
            if not vlo*(1-1e-12)<=vel<=vhi*(1+1e-12):return False
        fni=10.**(x[3]-x[0]-x[5]);lo,hi=self.config['bounds']['f_Ni']
        return bool(lo<=fni<=hi)

    def unpack(self,x):
        v=physical_samples(x)
        p=dict(M_ej=v[0],v_ej=np.sqrt(2*v[1]*1e51/(v[0]*M_SUN))/1e9,E_Th_in=v[2],M_ni=v[3],R_0=v[4],f_ni=v[5],T_floor=v[6],kappa_gamma=v[7],f_env=v[9],n=v[10],
               kappa=self.config['kappa'],q_core=self.config['q_core'],delta=0.,thermal_kind=0,event_grid=True,exact_resistance=True,
               output_step_days=self.config['output_step_days'],t_max_days=self.config['t_max_days'])
        return p,float(v[8])

    def predict(self,x,phase=None,band=None,nx=None,ny=None,return_state=False):
        if not self.valid(x):raise ValueError('Parameters outside physical support')
        phase=self.t if phase is None else np.asarray(phase,float)
        band=self.band if band is None else np.asarray(band)
        if phase.shape!=band.shape or phase.ndim!=1:raise ValueError('phase and band must be matching 1D arrays')
        if not set(band)<=set(self.table):raise ValueError('Requested band absent from configured data')
        p,dt=self.unpack(x);age=(phase-dt)/(1+self.config['redshift'])
        if np.any(age<=0) or np.any(age>=self.config['t_max_days']):raise ValueError('Prediction age outside transport interval')
        p['t_eval_days']=np.unique(np.r_[age,(self.t-dt)/(1+self.config['redshift'])])
        state,meta=solve(p,'structure',nx or self.config['nx'],ny or self.config['ny'])
        tp,rp,td,rd=_nickel_two_component_blackbody(state,p['T_floor']);result=np.empty(len(phase))
        for b in set(band):
            fp=10**(-.4*self.unit_magnitude(b,tp))*rp**2
            fd=10**(-.4*self.unit_magnitude(b,td))*rd**2
            model_mag=-2.5*np.log10(fp+fd);m=band==b
            result[m]=np.interp(age[m],state.t_s/DAY,model_mag,left=np.nan,right=np.nan)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite model magnitudes')
        return (result,meta,state) if return_state else (result,meta)

    def residual(self,x):
        if not self.valid(x):
            fni=10.**np.clip(x[3]-x[0]-x[5],-100,100)
            lo,hi=self.config['bounds']['f_Ni']
            return np.full(len(self.t),1e5*(1+max(fni-hi,0)+max(lo-fni,0)))
        try:return (self.predict(x)[0]-self.y)/self.sigma
        except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):return np.full(len(self.t),1e6)

    def prior_transform(self,u):
        """Uniform physical box prior, then transform to numerical coordinates.

        f_Ni is uniform, NOT xi. xi bounds are a hard physical support cut
        in log_likelihood. Reported evidence includes this cut's prior volume.
        """
        u=np.asarray(u,float)
        if u.shape!=(11,) or np.any(u<0) or np.any(u>1):raise ValueError('Invalid unit cube')
        bounds=np.array([self.config['bounds'][k] for k in PHYSICAL_NAMES])
        a=bounds[:,0]+u*(bounds[:,1]-bounds[:,0])
        if self.config['prior']=='velocity_uniform':
            vlo,vhi=self.config['velocity_prior_bounds'];v=vlo+u[1]*(vhi-vlo)
            a[1]=.5*a[0]*M_SUN*(v*1e9)**2/1e51
        a[5]=a[3]/(a[0]*a[5])
        a[LOG_INDICES]=np.log10(a[LOG_INDICES])
        return a

    def log_likelihood(self,x):
        # dynesty must receive ONLY likelihood, never the old log posterior.
        if not self.valid(x):return -np.inf
        try:r=(self.predict(x)[0]-self.y)/self.sigma
        except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):return -np.inf
        if not np.isfinite(r).all():return -np.inf
        return float(-.5*(r@r)-np.log(self.sigma).sum()-.5*len(r)*np.log(2*np.pi))

    def fingerprint(self):
        source=Path(__file__).read_text(encoding='utf-8')
        # Run controls (steps/resume/output/seed) must be editable without invalidating the model.
        marker_start='# === USER SETTINGS '+'BEGIN ==='
        marker_end='# === USER SETTINGS '+'END ==='
        before,rest=source.split(marker_start,1)
        _,after=rest.split(marker_end,1)
        science_source=before+after
        config=copy.deepcopy(self.config)
        config.pop('data_file',None);config.pop('name',None)
        payload={'config':config,'data_sha256':sha256(self.data_path),
                 'model_source_sha256':hashlib.sha256(science_source.encode()).hexdigest(),
                 'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','numba','dynesty']}}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

# === Optimization, dynamic nested sampling and export ===
WORKER_MODEL=None

def worker_init(config,nx,ny):
    global WORKER_MODEL
    WORKER_MODEL=Model(config,nx,ny)

def worker_log_likelihood(x):return WORKER_MODEL.log_likelihood(x)

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

@contextlib.contextmanager
def run_lock(directory):
    path=Path(directory)/'.run.lock'
    with path.open('a+b') as handle:
        try:
            if os.name=='nt':
                import msvcrt
                handle.seek(0,2)
                if handle.tell()==0:handle.write(b'0');handle.flush()
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('Another process is using this run directory') from exc
        try:yield
        finally:
            if os.name=='nt':handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)

def load_coordinates(path,default):
    if path is None:return np.array(default,float).copy()
    d=json.loads(Path(path).read_text())
    if d.get('coordinate_names')!=COORDINATES:raise ValueError('Parameter file schema differs; old velocity/emcee JSON cannot be reused directly')
    return np.asarray(d['sampling_coordinates'],float)

def fit_record(model,x,grid_check=False):
    y,meta=model.predict(x);r=(y-model.y)/model.sigma
    a=physical_samples(x);derived=derived_samples(x);fraction=(x-model.bounds[:,0])/(model.bounds[:,1]-model.bounds[:,0])
    result={'name':model.config['name'],'status':'Optimization/evaluation only; not posterior inference',
            'sampling_coordinates':np.asarray(x).tolist(),'coordinate_names':COORDINATES,
            'physical_parameters':dict(zip(PHYSICAL_NAMES,map(float,a))),
            'derived_parameters':dict(zip(DERIVED_NAMES,map(float,derived))),
            'chi2':float(r@r),'chi2_reduced':float(r@r)/(len(r)-11),'rms_mag':float(np.sqrt(np.mean((y-model.y)**2))),
            'n_points':len(r),'n_parameters':11,'grid':[model.config['nx'],model.config['ny']],
            'near_coordinate_bounds':[COORDINATES[i] for i in np.where((fraction<.005)|(fraction>.995))[0]],
            'meta':meta,'config':model.config}
    if grid_check:
        fine,_=model.predict(x,nx=model.config['nx']*2,ny=model.config['ny']*2)
        result['grid_check']={'max_delta_mag':float(np.max(abs(y-fine))),'rms_delta_mag':float(np.sqrt(np.mean((y-fine)**2)))}
    return result,y

def save_fit(model,x,out,grid_check=False,optimizer=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result,y=fit_record(model,x,grid_check)
    if optimizer is not None:result['optimizer']=optimizer
    atomic_json(out/'best_fit.json',result)
    with (out/'predictions_at_data.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['absolute_time','phase_days','band','observed_vega_mag','statistical_error_mag','used_error_mag','model_vega_mag','model_minus_data_mag'])
        w.writerows(zip(model.absolute_time,model.t,model.band,model.y,model.err,model.sigma,y,y-model.y))
    print(json.dumps({k:result[k] for k in ['chi2','chi2_reduced','rms_mag','physical_parameters']},ensure_ascii=False,indent=2),flush=True)
    return result

def fit(args,model):
    out=Path(args.out)
    if (out/'best_fit.json').exists():raise ValueError('Output already contains best_fit.json; choose a new --out')
    x=load_coordinates(args.initial,model.initial)
    if not model.valid(x):raise ValueError('Invalid initial coordinates')
    starts=[x.copy()];rng=np.random.default_rng(args.seed)
    for i in range(1,args.starts):
        z=x.copy()
        if i==1:z[4]=np.log10(7000);z[2]=0.;z[10]=10.
        elif i==2:z[4]=np.log10(500);z[2]=np.log10(80);z[10]=7.
        else:z+=rng.normal(size=11)*.04*(model.bounds[:,1]-model.bounds[:,0])
        z=np.clip(z,model.bounds[:,0]+1e-8,model.bounds[:,1]-1e-8)
        if model.valid(z):starts.append(z)
    out.mkdir(parents=True,exist_ok=True);best=x;best_chi=float(model.residual(x)@model.residual(x))
    for i,z in enumerate(starts):
        began=time.time()
        ls=least_squares(model.residual,z,bounds=(model.bounds[:,0],model.bounds[:,1]),max_nfev=args.max_nfev,diff_step=1e-4,ftol=1e-8,xtol=1e-9,gtol=1e-7,x_scale='jac')
        chi=float(ls.fun@ls.fun)
        record,_=fit_record(model,ls.x)
        record['optimizer']={'success':bool(ls.success),'message':str(ls.message),'nfev':ls.nfev,'optimality':float(ls.optimality),'elapsed_seconds':time.time()-began}
        atomic_json(out/f'start_{i}.json',record)
        if chi<best_chi:best=ls.x.copy();best_chi=chi
        print(f'start {i}: chi2={chi:.4f}, R0={10**ls.x[4]:.3f} Rsun, n={ls.x[10]:.4f}',flush=True)
    save_fit(model,best,out,args.grid_check,{'starts_completed':len(starts),'note':'Best found point, not a proof of global optimality'})

def export_nested(sampler,model,out,status):
    results=sampler.results
    coords=np.asarray(results.samples)
    logwt=np.asarray(results.logwt)
    if not len(coords) or not np.isfinite(logsumexp(logwt)):
        print('No finite posterior weights yet; use checkpoint to resume.',flush=True)
        return
    weights=np.exp(logwt-logsumexp(logwt))
    physical=physical_samples(coords);derived=derived_samples(coords)
    np.savez_compressed(out/'nested_results.npz',sampling_coordinates=coords,
        physical_samples=physical,derived_samples=derived,weights=weights,
        log_likelihood=results.logl,logwt=logwt,logz=results.logz,
        logzerr=results.logzerr,physical_names=np.array(PHYSICAL_NAMES),
        coordinate_names=np.array(COORDINATES),derived_names=np.array(DERIVED_NAMES))
    equal=dyfunc.resample_equal(physical,weights,rstate=np.random.default_rng(RANDOM_SEED))
    np.save(out/'posterior_physical_equal.npy',equal,allow_pickle=False)
    qs=np.array([dyfunc.quantile(physical[:,i],[.16,.5,.84],weights=weights) for i in range(11)])
    atomic_json(out/'quantiles.json',{n:dict(zip(['p16','median','p84'],map(float,qs[i]))) for i,n in enumerate(PHYSICAL_NAMES)})
    i=int(np.argmax(results.logl))
    atomic_json(out/'maximum_likelihood_sample.json',{
        'coordinate_names':COORDINATES,'sampling_coordinates':coords[i].tolist(),
        'physical_parameters':dict(zip(PHYSICAL_NAMES,map(float,physical[i]))),
        'log_likelihood':float(results.logl[i]),'note':'Maximum likelihood sample; not marginal medians or a MAP claim'})
    atomic_json(out/'diagnostics.json',{'status':status,'samples':len(coords),
        'likelihood_calls':int(np.sum(results.ncall)),
        'weight_ESS':float(1/np.sum(weights**2)),
        'logz':float(results.logz[-1]),'logzerr':float(results.logzerr[-1]),
        'note':'Weighted nested samples: no burn-in/thinning. ESS or stopping alone does not prove modes were all found.',
        'evidence_note':'Evidence uses the physical box prior with zero likelihood outside xi support; not normalized to the xi-truncated prior.'})
    print(f'Exported weighted posterior and equal-weight physical samples to {out}',flush=True)


def nested(args,model):
    global WORKER_MODEL
    WORKER_MODEL=model  # Also supports resuming a pooled checkpoint with one worker.
    if N_WORKERS<1 or NLIVE_INIT<22 or NLIVE_BATCH<22:
        raise ValueError('N_WORKERS>=1 and live point counts>=22 required')
    if DLOGZ_INIT<=0 or N_EFFECTIVE<1 or CHECKPOINT_EVERY<=0:
        raise ValueError('Invalid stopping/checkpoint settings')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        checkpoint=out/'checkpoint.pkl';meta_path=out/'run_metadata.json'
        signature,details=model.fingerprint()
        sampler_settings={'nlive_init':NLIVE_INIT,'nlive_batch':NLIVE_BATCH,
            'bound':'multi','sample':'rwalk','walks':30}
        if RESUME:
            if not checkpoint.exists() or not meta_path.exists():raise FileNotFoundError('Resume needs checkpoint.pkl and run_metadata.json')
            metadata=json.loads(meta_path.read_text())
            if metadata['fingerprint']!=signature or metadata['sampler_settings']!=sampler_settings:
                raise ValueError('Resume refused: model/data/prior/versions or sampler settings changed; use a new RESULT_DIR')
            if metadata['status'].startswith('finished'):
                raise ValueError('This run is already finished. Use a new RESULT_DIR for an independent run.')
        else:
            if checkpoint.exists() or meta_path.exists():raise ValueError('Run exists; use RESUME=True or a new RESULT_DIR')
            metadata={'fingerprint':signature,'fingerprint_details':details,
                'resolved_config':model.config,'coordinate_names':COORDINATES,
                'physical_names':PHYSICAL_NAMES,'derived_names':DERIVED_NAMES,
                'sampler':'dynesty.DynamicNestedSampler','sampler_settings':sampler_settings,
                'seed':RANDOM_SEED,'status':'running',
                'time_definition':'rest_age=(JD-reference_epoch-tshift)/(1+z)',
                'energy_definition':'E_k_1e51_erg = 0.5 * Mej_g * v_ej_cms**2 / 1e51',
                'prior_note':'Independent physical uniform priors in all other parameters, including f_Ni; xi bounds enforced as zero likelihood. Velocity-uniform and energy-uniform differ.'}
        metadata['status']='running';atomic_json(meta_path,metadata)
        # Warm up transport before launching workers.
        model.predict(model.initial)
        with contextlib.ExitStack() as stack:
            pool=None
            if N_WORKERS>1:
                pool=stack.enter_context(mp.get_context('spawn').Pool(N_WORKERS,
                    initializer=worker_init,initargs=(model.config,model.config['nx'],model.config['ny'])))
            if RESUME:
                sampler=dynesty.DynamicNestedSampler.restore(str(checkpoint),pool=pool)
            else:
                fn=worker_log_likelihood if pool is not None else model.log_likelihood
                sampler=dynesty.DynamicNestedSampler(fn,model.prior_transform,11,
                    nlive=NLIVE_INIT,bound='multi',sample='rwalk',walks=30,
                    rstate=np.random.default_rng(RANDOM_SEED),pool=pool,
                    queue_size=N_WORKERS if pool is not None else None,
                    use_pool={'prior_transform':False,'loglikelihood':True})
            status='finished'
            try:
                sampler.run_nested(nlive_init=NLIVE_INIT,nlive_batch=NLIVE_BATCH,
                    dlogz_init=DLOGZ_INIT,n_effective=N_EFFECTIVE,
                    maxcall=MAXCALL,maxiter=MAXITER,maxbatch=MAXBATCH,
                    checkpoint_file=str(checkpoint),checkpoint_every=CHECKPOINT_EVERY,
                    resume=RESUME,print_progress=True)
                if any(v is not None for v in (MAXCALL,MAXITER,MAXBATCH)):
                    status='finished_with_user_limits_check_diagnostics'
            except KeyboardInterrupt:
                status='interrupted'
                print('Interrupted; saving a checkpoint and available weighted samples.',flush=True)
            finally:
                # Preserve unexpected exceptions as errors rather than claiming completion.
                if sys.exc_info()[0] is not None:status='error'
                sampler.save(str(checkpoint))
                metadata['status']=status;atomic_json(meta_path,metadata)
            export_nested(sampler,model,out,status)


def predict_curve(args,model):
    x=load_coordinates(args.initial,model.initial)
    phase=np.linspace(model.t.min(),model.t.max(),PREDICTION_POINTS)
    b=np.tile(model.config['bands'],len(phase));t=np.repeat(phase,len(model.config['bands']))
    y,_=model.predict(x,t,b)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['phase_days','band','model_vega_mag']);w.writerows(zip(t,b,y))


def main():
    if RUN_MODE not in ('nested','evaluate','fit','predict'):raise ValueError('Unknown RUN_MODE')
    if int(dynesty.__version__.split('.')[0])!=3:raise RuntimeError('Use dynesty>=3.0,<4; tested with 3.1.0')
    root=Path(RESULT_DIR).expanduser().resolve()
    model=Model(MODEL_CONFIG)
    print(f'11 parameters: {PHYSICAL_NAMES}',flush=True)
    print(f'Prior={ENERGY_PRIOR}; {len(model.t)} data points; output={root}',flush=True)
    args=SimpleNamespace(initial=PARAMETER_FILE,seed=RANDOM_SEED,starts=FIT_STARTS,
        max_nfev=FIT_MAX_NFEV,grid_check=CHECK_FINER_GRID)
    if RUN_MODE=='nested':
        if PARAMETER_FILE is not None:raise ValueError('Nested live points are drawn from the full prior; set PARAMETER_FILE=None')
        args.out=root/'nested';nested(args,model)
    elif RUN_MODE=='evaluate':
        save_fit(model,load_coordinates(PARAMETER_FILE,model.initial),root/'evaluation',CHECK_FINER_GRID)
    elif RUN_MODE=='fit':
        args.out=root/'fit';fit(args,model)
    else:
        args.out=root/'model_curve.csv';predict_curve(args,model)

# 输出列：Mej(Msun),Ek(1e51erg),Eth(1e49erg),MNi(Msun),R0(Rsun),f_Ni,
# T_floor(K),kappa_gamma(cm2/g),tshift(day),f_env,n。
# 原传输求解器内部仍通过v=sqrt(2Ek/Mej)计算，不增加自由参数。
# tshift>0意味着爆炸晚于reference_epoch；本版保留[-1,+1]天先验。
# T_floor为有效颜色温度下限；kappa_gamma为有效逃逸描述。
# 默认κ=.07，qcore=1/3，初始u∝1/r；没有经验SED修正。
# nested_results.npz里的样本有weights；corner需要weights或用等权重npy。
# 旧emcee chain.h5与旧绘图notebook不能直接用于本版本，不做burn/thin。
# 初始参考点仅用于evaluate/fit及编译预热，dynesty从全先验初始化。
# 修改采样器和参数化不会自动消除原模型的系统残差或物理简化。

GPL_LICENSE_TEXT = '                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n\n Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n Everyone is permitted to copy and distribute verbatim copies\n of this license document, but changing it is not allowed.\n\n                            Preamble\n\n  The GNU General Public License is a free, copyleft license for\nsoftware and other kinds of works.\n\n  The licenses for most software and other practical works are designed\nto take away your freedom to share and change the works.  By contrast,\nthe GNU General Public License is intended to guarantee your freedom to\nshare and change all versions of a program--to make sure it remains free\nsoftware for all its users.  We, the Free Software Foundation, use the\nGNU General Public License for most of our software; it applies also to\nany other work released this way by its authors.  You can apply it to\nyour programs, too.\n\n  When we speak of free software, we are referring to freedom, not\nprice.  Our General Public Licenses are designed to make sure that you\nhave the freedom to distribute copies of free software (and charge for\nthem if you wish), that you receive source code or can get it if you\nwant it, that you can change the software or use pieces of it in new\nfree programs, and that you know you can do these things.\n\n  To protect your rights, we need to prevent others from denying you\nthese rights or asking you to surrender the rights.  Therefore, you have\ncertain responsibilities if you distribute copies of the software, or if\nyou modify it: responsibilities to respect the freedom of others.\n\n  For example, if you distribute copies of such a program, whether\ngratis or for a fee, you must pass on to the recipients the same\nfreedoms that you received.  You must make sure that they, too, receive\nor can get the source code.  And you must show them these terms so they\nknow their rights.\n\n  Developers that use the GNU GPL protect your rights with two steps:\n(1) assert copyright on the software, and (2) offer you this License\ngiving you legal permission to copy, distribute and/or modify it.\n\n  For the developers\' and authors\' protection, the GPL clearly explains\nthat there is no warranty for this free software.  For both users\' and\nauthors\' sake, the GPL requires that modified versions be marked as\nchanged, so that their problems will not be attributed erroneously to\nauthors of previous versions.\n\n  Some devices are designed to deny users access to install or run\nmodified versions of the software inside them, although the manufacturer\ncan do so.  This is fundamentally incompatible with the aim of\nprotecting users\' freedom to change the software.  The systematic\npattern of such abuse occurs in the area of products for individuals to\nuse, which is precisely where it is most unacceptable.  Therefore, we\nhave designed this version of the GPL to prohibit the practice for those\nproducts.  If such problems arise substantially in other domains, we\nstand ready to extend this provision to those domains in future versions\nof the GPL, as needed to protect the freedom of users.\n\n  Finally, every program is threatened constantly by software patents.\nStates should not allow patents to restrict development and use of\nsoftware on general-purpose computers, but in those that do, we wish to\navoid the special danger that patents applied to a free program could\nmake it effectively proprietary.  To prevent this, the GPL assures that\npatents cannot be used to render the program non-free.\n\n  The precise terms and conditions for copying, distribution and\nmodification follow.\n\n                       TERMS AND CONDITIONS\n\n  0. Definitions.\n\n  "This License" refers to version 3 of the GNU General Public License.\n\n  "Copyright" also means copyright-like laws that apply to other kinds of\nworks, such as semiconductor masks.\n\n  "The Program" refers to any copyrightable work licensed under this\nLicense.  Each licensee is addressed as "you".  "Licensees" and\n"recipients" may be individuals or organizations.\n\n  To "modify" a work means to copy from or adapt all or part of the work\nin a fashion requiring copyright permission, other than the making of an\nexact copy.  The resulting work is called a "modified version" of the\nearlier work or a work "based on" the earlier work.\n\n  A "covered work" means either the unmodified Program or a work based\non the Program.\n\n  To "propagate" a work means to do anything with it that, without\npermission, would make you directly or secondarily liable for\ninfringement under applicable copyright law, except executing it on a\ncomputer or modifying a private copy.  Propagation includes copying,\ndistribution (with or without modification), making available to the\npublic, and in some countries other activities as well.\n\n  To "convey" a work means any kind of propagation that enables other\nparties to make or receive copies.  Mere interaction with a user through\na computer network, with no transfer of a copy, is not conveying.\n\n  An interactive user interface displays "Appropriate Legal Notices"\nto the extent that it includes a convenient and prominently visible\nfeature that (1) displays an appropriate copyright notice, and (2)\ntells the user that there is no warranty for the work (except to the\nextent that warranties are provided), that licensees may convey the\nwork under this License, and how to view a copy of this License.  If\nthe interface presents a list of user commands or options, such as a\nmenu, a prominent item in the list meets this criterion.\n\n  1. Source Code.\n\n  The "source code" for a work means the preferred form of the work\nfor making modifications to it.  "Object code" means any non-source\nform of a work.\n\n  A "Standard Interface" means an interface that either is an official\nstandard defined by a recognized standards body, or, in the case of\ninterfaces specified for a particular programming language, one that\nis widely used among developers working in that language.\n\n  The "System Libraries" of an executable work include anything, other\nthan the work as a whole, that (a) is included in the normal form of\npackaging a Major Component, but which is not part of that Major\nComponent, and (b) serves only to enable use of the work with that\nMajor Component, or to implement a Standard Interface for which an\nimplementation is available to the public in source code form.  A\n"Major Component", in this context, means a major essential component\n(kernel, window system, and so on) of the specific operating system\n(if any) on which the executable work runs, or a compiler used to\nproduce the work, or an object code interpreter used to run it.\n\n  The "Corresponding Source" for a work in object code form means all\nthe source code needed to generate, install, and (for an executable\nwork) run the object code and to modify the work, including scripts to\ncontrol those activities.  However, it does not include the work\'s\nSystem Libraries, or general-purpose tools or generally available free\nprograms which are used unmodified in performing those activities but\nwhich are not part of the work.  For example, Corresponding Source\nincludes interface definition files associated with source files for\nthe work, and the source code for shared libraries and dynamically\nlinked subprograms that the work is specifically designed to require,\nsuch as by intimate data communication or control flow between those\nsubprograms and other parts of the work.\n\n  The Corresponding Source need not include anything that users\ncan regenerate automatically from other parts of the Corresponding\nSource.\n\n  The Corresponding Source for a work in source code form is that\nsame work.\n\n  2. Basic Permissions.\n\n  All rights granted under this License are granted for the term of\ncopyright on the Program, and are irrevocable provided the stated\nconditions are met.  This License explicitly affirms your unlimited\npermission to run the unmodified Program.  The output from running a\ncovered work is covered by this License only if the output, given its\ncontent, constitutes a covered work.  This License acknowledges your\nrights of fair use or other equivalent, as provided by copyright law.\n\n  You may make, run and propagate covered works that you do not\nconvey, without conditions so long as your license otherwise remains\nin force.  You may convey covered works to others for the sole purpose\nof having them make modifications exclusively for you, or provide you\nwith facilities for running those works, provided that you comply with\nthe terms of this License in conveying all material for which you do\nnot control copyright.  Those thus making or running the covered works\nfor you must do so exclusively on your behalf, under your direction\nand control, on terms that prohibit them from making any copies of\nyour copyrighted material outside their relationship with you.\n\n  Conveying under any other circumstances is permitted solely under\nthe conditions stated below.  Sublicensing is not allowed; section 10\nmakes it unnecessary.\n\n  3. Protecting Users\' Legal Rights From Anti-Circumvention Law.\n\n  No covered work shall be deemed part of an effective technological\nmeasure under any applicable law fulfilling obligations under article\n11 of the WIPO copyright treaty adopted on 20 December 1996, or\nsimilar laws prohibiting or restricting circumvention of such\nmeasures.\n\n  When you convey a covered work, you waive any legal power to forbid\ncircumvention of technological measures to the extent such circumvention\nis effected by exercising rights under this License with respect to\nthe covered work, and you disclaim any intention to limit operation or\nmodification of the work as a means of enforcing, against the work\'s\nusers, your or third parties\' legal rights to forbid circumvention of\ntechnological measures.\n\n  4. Conveying Verbatim Copies.\n\n  You may convey verbatim copies of the Program\'s source code as you\nreceive it, in any medium, provided that you conspicuously and\nappropriately publish on each copy an appropriate copyright notice;\nkeep intact all notices stating that this License and any\nnon-permissive terms added in accord with section 7 apply to the code;\nkeep intact all notices of the absence of any warranty; and give all\nrecipients a copy of this License along with the Program.\n\n  You may charge any price or no price for each copy that you convey,\nand you may offer support or warranty protection for a fee.\n\n  5. Conveying Modified Source Versions.\n\n  You may convey a work based on the Program, or the modifications to\nproduce it from the Program, in the form of source code under the\nterms of section 4, provided that you also meet all of these conditions:\n\n    a) The work must carry prominent notices stating that you modified\n    it, and giving a relevant date.\n\n    b) The work must carry prominent notices stating that it is\n    released under this License and any conditions added under section\n    7.  This requirement modifies the requirement in section 4 to\n    "keep intact all notices".\n\n    c) You must license the entire work, as a whole, under this\n    License to anyone who comes into possession of a copy.  This\n    License will therefore apply, along with any applicable section 7\n    additional terms, to the whole of the work, and all its parts,\n    regardless of how they are packaged.  This License gives no\n    permission to license the work in any other way, but it does not\n    invalidate such permission if you have separately received it.\n\n    d) If the work has interactive user interfaces, each must display\n    Appropriate Legal Notices; however, if the Program has interactive\n    interfaces that do not display Appropriate Legal Notices, your\n    work need not make them do so.\n\n  A compilation of a covered work with other separate and independent\nworks, which are not by their nature extensions of the covered work,\nand which are not combined with it such as to form a larger program,\nin or on a volume of a storage or distribution medium, is called an\n"aggregate" if the compilation and its resulting copyright are not\nused to limit the access or legal rights of the compilation\'s users\nbeyond what the individual works permit.  Inclusion of a covered work\nin an aggregate does not cause this License to apply to the other\nparts of the aggregate.\n\n  6. Conveying Non-Source Forms.\n\n  You may convey a covered work in object code form under the terms\nof sections 4 and 5, provided that you also convey the\nmachine-readable Corresponding Source under the terms of this License,\nin one of these ways:\n\n    a) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by the\n    Corresponding Source fixed on a durable physical medium\n    customarily used for software interchange.\n\n    b) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by a\n    written offer, valid for at least three years and valid for as\n    long as you offer spare parts or customer support for that product\n    model, to give anyone who possesses the object code either (1) a\n    copy of the Corresponding Source for all the software in the\n    product that is covered by this License, on a durable physical\n    medium customarily used for software interchange, for a price no\n    more than your reasonable cost of physically performing this\n    conveying of source, or (2) access to copy the\n    Corresponding Source from a network server at no charge.\n\n    c) Convey individual copies of the object code with a copy of the\n    written offer to provide the Corresponding Source.  This\n    alternative is allowed only occasionally and noncommercially, and\n    only if you received the object code with such an offer, in accord\n    with subsection 6b.\n\n    d) Convey the object code by offering access from a designated\n    place (gratis or for a charge), and offer equivalent access to the\n    Corresponding Source in the same way through the same place at no\n    further charge.  You need not require recipients to copy the\n    Corresponding Source along with the object code.  If the place to\n    copy the object code is a network server, the Corresponding Source\n    may be on a different server (operated by you or a third party)\n    that supports equivalent copying facilities, provided you maintain\n    clear directions next to the object code saying where to find the\n    Corresponding Source.  Regardless of what server hosts the\n    Corresponding Source, you remain obligated to ensure that it is\n    available for as long as needed to satisfy these requirements.\n\n    e) Convey the object code using peer-to-peer transmission, provided\n    you inform other peers where the object code and Corresponding\n    Source of the work are being offered to the general public at no\n    charge under subsection 6d.\n\n  A separable portion of the object code, whose source code is excluded\nfrom the Corresponding Source as a System Library, need not be\nincluded in conveying the object code work.\n\n  A "User Product" is either (1) a "consumer product", which means any\ntangible personal property which is normally used for personal, family,\nor household purposes, or (2) anything designed or sold for incorporation\ninto a dwelling.  In determining whether a product is a consumer product,\ndoubtful cases shall be resolved in favor of coverage.  For a particular\nproduct received by a particular user, "normally used" refers to a\ntypical or common use of that class of product, regardless of the status\nof the particular user or of the way in which the particular user\nactually uses, or expects or is expected to use, the product.  A product\nis a consumer product regardless of whether the product has substantial\ncommercial, industrial or non-consumer uses, unless such uses represent\nthe only significant mode of use of the product.\n\n  "Installation Information" for a User Product means any methods,\nprocedures, authorization keys, or other information required to install\nand execute modified versions of a covered work in that User Product from\na modified version of its Corresponding Source.  The information must\nsuffice to ensure that the continued functioning of the modified object\ncode is in no case prevented or interfered with solely because\nmodification has been made.\n\n  If you convey an object code work under this section in, or with, or\nspecifically for use in, a User Product, and the conveying occurs as\npart of a transaction in which the right of possession and use of the\nUser Product is transferred to the recipient in perpetuity or for a\nfixed term (regardless of how the transaction is characterized), the\nCorresponding Source conveyed under this section must be accompanied\nby the Installation Information.  But this requirement does not apply\nif neither you nor any third party retains the ability to install\nmodified object code on the User Product (for example, the work has\nbeen installed in ROM).\n\n  The requirement to provide Installation Information does not include a\nrequirement to continue to provide support service, warranty, or updates\nfor a work that has been modified or installed by the recipient, or for\nthe User Product in which it has been modified or installed.  Access to a\nnetwork may be denied when the modification itself materially and\nadversely affects the operation of the network or violates the rules and\nprotocols for communication across the network.\n\n  Corresponding Source conveyed, and Installation Information provided,\nin accord with this section must be in a format that is publicly\ndocumented (and with an implementation available to the public in\nsource code form), and must require no special password or key for\nunpacking, reading or copying.\n\n  7. Additional Terms.\n\n  "Additional permissions" are terms that supplement the terms of this\nLicense by making exceptions from one or more of its conditions.\nAdditional permissions that are applicable to the entire Program shall\nbe treated as though they were included in this License, to the extent\nthat they are valid under applicable law.  If additional permissions\napply only to part of the Program, that part may be used separately\nunder those permissions, but the entire Program remains governed by\nthis License without regard to the additional permissions.\n\n  When you convey a copy of a covered work, you may at your option\nremove any additional permissions from that copy, or from any part of\nit.  (Additional permissions may be written to require their own\nremoval in certain cases when you modify the work.)  You may place\nadditional permissions on material, added by you to a covered work,\nfor which you have or can give appropriate copyright permission.\n\n  Notwithstanding any other provision of this License, for material you\nadd to a covered work, you may (if authorized by the copyright holders of\nthat material) supplement the terms of this License with terms:\n\n    a) Disclaiming warranty or limiting liability differently from the\n    terms of sections 15 and 16 of this License; or\n\n    b) Requiring preservation of specified reasonable legal notices or\n    author attributions in that material or in the Appropriate Legal\n    Notices displayed by works containing it; or\n\n    c) Prohibiting misrepresentation of the origin of that material, or\n    requiring that modified versions of such material be marked in\n    reasonable ways as different from the original version; or\n\n    d) Limiting the use for publicity purposes of names of licensors or\n    authors of the material; or\n\n    e) Declining to grant rights under trademark law for use of some\n    trade names, trademarks, or service marks; or\n\n    f) Requiring indemnification of licensors and authors of that\n    material by anyone who conveys the material (or modified versions of\n    it) with contractual assumptions of liability to the recipient, for\n    any liability that these contractual assumptions directly impose on\n    those licensors and authors.\n\n  All other non-permissive additional terms are considered "further\nrestrictions" within the meaning of section 10.  If the Program as you\nreceived it, or any part of it, contains a notice stating that it is\ngoverned by this License along with a term that is a further\nrestriction, you may remove that term.  If a license document contains\na further restriction but permits relicensing or conveying under this\nLicense, you may add to a covered work material governed by the terms\nof that license document, provided that the further restriction does\nnot survive such relicensing or conveying.\n\n  If you add terms to a covered work in accord with this section, you\nmust place, in the relevant source files, a statement of the\nadditional terms that apply to those files, or a notice indicating\nwhere to find the applicable terms.\n\n  Additional terms, permissive or non-permissive, may be stated in the\nform of a separately written license, or stated as exceptions;\nthe above requirements apply either way.\n\n  8. Termination.\n\n  You may not propagate or modify a covered work except as expressly\nprovided under this License.  Any attempt otherwise to propagate or\nmodify it is void, and will automatically terminate your rights under\nthis License (including any patent licenses granted under the third\nparagraph of section 11).\n\n  However, if you cease all violation of this License, then your\nlicense from a particular copyright holder is reinstated (a)\nprovisionally, unless and until the copyright holder explicitly and\nfinally terminates your license, and (b) permanently, if the copyright\nholder fails to notify you of the violation by some reasonable means\nprior to 60 days after the cessation.\n\n  Moreover, your license from a particular copyright holder is\nreinstated permanently if the copyright holder notifies you of the\nviolation by some reasonable means, this is the first time you have\nreceived notice of violation of this License (for any work) from that\ncopyright holder, and you cure the violation prior to 30 days after\nyour receipt of the notice.\n\n  Termination of your rights under this section does not terminate the\nlicenses of parties who have received copies or rights from you under\nthis License.  If your rights have been terminated and not permanently\nreinstated, you do not qualify to receive new licenses for the same\nmaterial under section 10.\n\n  9. Acceptance Not Required for Having Copies.\n\n  You are not required to accept this License in order to receive or\nrun a copy of the Program.  Ancillary propagation of a covered work\noccurring solely as a consequence of using peer-to-peer transmission\nto receive a copy likewise does not require acceptance.  However,\nnothing other than this License grants you permission to propagate or\nmodify any covered work.  These actions infringe copyright if you do\nnot accept this License.  Therefore, by modifying or propagating a\ncovered work, you indicate your acceptance of this License to do so.\n\n  10. Automatic Licensing of Downstream Recipients.\n\n  Each time you convey a covered work, the recipient automatically\nreceives a license from the original licensors, to run, modify and\npropagate that work, subject to this License.  You are not responsible\nfor enforcing compliance by third parties with this License.\n\n  An "entity transaction" is a transaction transferring control of an\norganization, or substantially all assets of one, or subdividing an\norganization, or merging organizations.  If propagation of a covered\nwork results from an entity transaction, each party to that\ntransaction who receives a copy of the work also receives whatever\nlicenses to the work the party\'s predecessor in interest had or could\ngive under the previous paragraph, plus a right to possession of the\nCorresponding Source of the work from the predecessor in interest, if\nthe predecessor has it or can get it with reasonable efforts.\n\n  You may not impose any further restrictions on the exercise of the\nrights granted or affirmed under this License.  For example, you may\nnot impose a license fee, royalty, or other charge for exercise of\nrights granted under this License, and you may not initiate litigation\n(including a cross-claim or counterclaim in a lawsuit) alleging that\nany patent claim is infringed by making, using, selling, offering for\nsale, or importing the Program or any portion of it.\n\n  11. Patents.\n\n  A "contributor" is a copyright holder who authorizes use under this\nLicense of the Program or a work on which the Program is based.  The\nwork thus licensed is called the contributor\'s "contributor version".\n\n  A contributor\'s "essential patent claims" are all patent claims\nowned or controlled by the contributor, whether already acquired or\nhereafter acquired, that would be infringed by some manner, permitted\nby this License, of making, using, or selling its contributor version,\nbut do not include claims that would be infringed only as a\nconsequence of further modification of the contributor version.  For\npurposes of this definition, "control" includes the right to grant\npatent sublicenses in a manner consistent with the requirements of\nthis License.\n\n  Each contributor grants you a non-exclusive, worldwide, royalty-free\npatent license under the contributor\'s essential patent claims, to\nmake, use, sell, offer for sale, import and otherwise run, modify and\npropagate the contents of its contributor version.\n\n  In the following three paragraphs, a "patent license" is any express\nagreement or commitment, however denominated, not to enforce a patent\n(such as an express permission to practice a patent or covenant not to\nsue for patent infringement).  To "grant" such a patent license to a\nparty means to make such an agreement or commitment not to enforce a\npatent against the party.\n\n  If you convey a covered work, knowingly relying on a patent license,\nand the Corresponding Source of the work is not available for anyone\nto copy, free of charge and under the terms of this License, through a\npublicly available network server or other readily accessible means,\nthen you must either (1) cause the Corresponding Source to be so\navailable, or (2) arrange to deprive yourself of the benefit of the\npatent license for this particular work, or (3) arrange, in a manner\nconsistent with the requirements of this License, to extend the patent\nlicense to downstream recipients.  "Knowingly relying" means you have\nactual knowledge that, but for the patent license, your conveying the\ncovered work in a country, or your recipient\'s use of the covered work\nin a country, would infringe one or more identifiable patents in that\ncountry that you have reason to believe are valid.\n\n  If, pursuant to or in connection with a single transaction or\narrangement, you convey, or propagate by procuring conveyance of, a\ncovered work, and grant a patent license to some of the parties\nreceiving the covered work authorizing them to use, propagate, modify\nor convey a specific copy of the covered work, then the patent license\nyou grant is automatically extended to all recipients of the covered\nwork and works based on it.\n\n  A patent license is "discriminatory" if it does not include within\nthe scope of its coverage, prohibits the exercise of, or is\nconditioned on the non-exercise of one or more of the rights that are\nspecifically granted under this License.  You may not convey a covered\nwork if you are a party to an arrangement with a third party that is\nin the business of distributing software, under which you make payment\nto the third party based on the extent of your activity of conveying\nthe work, and under which the third party grants, to any of the\nparties who would receive the covered work from you, a discriminatory\npatent license (a) in connection with copies of the covered work\nconveyed by you (or copies made from those copies), or (b) primarily\nfor and in connection with specific products or compilations that\ncontain the covered work, unless you entered into that arrangement,\nor that patent license was granted, prior to 28 March 2007.\n\n  Nothing in this License shall be construed as excluding or limiting\nany implied license or other defenses to infringement that may\notherwise be available to you under applicable patent law.\n\n  12. No Surrender of Others\' Freedom.\n\n  If conditions are imposed on you (whether by court order, agreement or\notherwise) that contradict the conditions of this License, they do not\nexcuse you from the conditions of this License.  If you cannot convey a\ncovered work so as to satisfy simultaneously your obligations under this\nLicense and any other pertinent obligations, then as a consequence you may\nnot convey it at all.  For example, if you agree to terms that obligate you\nto collect a royalty for further conveying from those to whom you convey\nthe Program, the only way you could satisfy both those terms and this\nLicense would be to refrain entirely from conveying the Program.\n\n  13. Use with the GNU Affero General Public License.\n\n  Notwithstanding any other provision of this License, you have\npermission to link or combine any covered work with a work licensed\nunder version 3 of the GNU Affero General Public License into a single\ncombined work, and to convey the resulting work.  The terms of this\nLicense will continue to apply to the part which is the covered work,\nbut the special requirements of the GNU Affero General Public License,\nsection 13, concerning interaction through a network will apply to the\ncombination as such.\n\n  14. Revised Versions of this License.\n\n  The Free Software Foundation may publish revised and/or new versions of\nthe GNU General Public License from time to time.  Such new versions will\nbe similar in spirit to the present version, but may differ in detail to\naddress new problems or concerns.\n\n  Each version is given a distinguishing version number.  If the\nProgram specifies that a certain numbered version of the GNU General\nPublic License "or any later version" applies to it, you have the\noption of following the terms and conditions either of that numbered\nversion or of any later version published by the Free Software\nFoundation.  If the Program does not specify a version number of the\nGNU General Public License, you may choose any version ever published\nby the Free Software Foundation.\n\n  If the Program specifies that a proxy can decide which future\nversions of the GNU General Public License can be used, that proxy\'s\npublic statement of acceptance of a version permanently authorizes you\nto choose that version for the Program.\n\n  Later license versions may give you additional or different\npermissions.  However, no additional obligations are imposed on any\nauthor or copyright holder as a result of your choosing to follow a\nlater version.\n\n  15. Disclaimer of Warranty.\n\n  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY\nAPPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT\nHOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY\nOF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,\nTHE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR\nPURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM\nIS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF\nALL NECESSARY SERVICING, REPAIR OR CORRECTION.\n\n  16. Limitation of Liability.\n\n  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING\nWILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS\nTHE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY\nGENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE\nUSE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF\nDATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD\nPARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),\nEVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF\nSUCH DAMAGES.\n\n  17. Interpretation of Sections 15 and 16.\n\n  If the disclaimer of warranty and limitation of liability provided\nabove cannot be given local legal effect according to their terms,\nreviewing courts shall apply local law that most closely approximates\nan absolute waiver of all civil liability in connection with the\nProgram, unless a warranty or assumption of liability accompanies a\ncopy of the Program in return for a fee.\n\n                     END OF TERMS AND CONDITIONS\n\n            How to Apply These Terms to Your New Programs\n\n  If you develop a new program, and you want it to be of the greatest\npossible use to the public, the best way to achieve this is to make it\nfree software which everyone can redistribute and change under these terms.\n\n  To do so, attach the following notices to the program.  It is safest\nto attach them to the start of each source file to most effectively\nstate the exclusion of warranty; and each file should have at least\nthe "copyright" line and a pointer to where the full notice is found.\n\n    <one line to give the program\'s name and a brief idea of what it does.>\n    Copyright (C) <year>  <name of author>\n\n    This program is free software: you can redistribute it and/or modify\n    it under the terms of the GNU General Public License as published by\n    the Free Software Foundation, either version 3 of the License, or\n    (at your option) any later version.\n\n    This program is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n    GNU General Public License for more details.\n\n    You should have received a copy of the GNU General Public License\n    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\nAlso add information on how to contact you by electronic and paper mail.\n\n  If the program does terminal interaction, make it output a short\nnotice like this when it starts in an interactive mode:\n\n    <program>  Copyright (C) <year>  <name of author>\n    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w\'.\n    This is free software, and you are welcome to redistribute it\n    under certain conditions; type `show c\' for details.\n\nThe hypothetical commands `show w\' and `show c\' should show the appropriate\nparts of the General Public License.  Of course, your program\'s commands\nmight be different; for a GUI interface, you would use an "about box".\n\n  You should also get your employer (if you work as a programmer) or school,\nif any, to sign a "copyright disclaimer" for the program, if necessary.\nFor more information on this, and how to apply and follow the GNU GPL, see\n<https://www.gnu.org/licenses/>.\n\n  The GNU General Public License does not permit incorporating your program\ninto proprietary programs.  If your program is a subroutine library, you\nmay consider it more useful to permit linking proprietary applications with\nthe library.  If this is what you want to do, use the GNU Lesser General\nPublic License instead of this License.  But first, please read\n<https://www.gnu.org/licenses/why-not-lgpl.html>.\n\n'


if __name__=='__main__':
    mp.freeze_support()
    main()
