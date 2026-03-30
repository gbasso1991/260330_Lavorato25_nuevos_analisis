#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
import matplotlib as mpl

#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz
# OH
resultados_OH = glob("OH_en_H2O/**/**/*resultados.txt")
resultados_OH.sort()
conc_OH = 4.09 #g/L

# CP hs
resultados_CP = glob("CP_en_VS55/**/**/*resultados.txt")
resultados_CP.sort()
conc_CP = 1.62 #g/L

print('\n')
for res in resultados_OH:
    print('  ',res)
for res in resultados_CP:
    print('  ',res)
#%% Listas con Resultados
res_OH=[]
print('Resultados OH hs', '='*80,'\n')
for r in resultados_OH:
    res_OH.append(ResultadosESAR(os.path.dirname(r)))

res_CP=[]
print('Resultados CP hs', '='*80,'\n')
for r in resultados_CP:
    res_CP.append(ResultadosESAR(os.path.dirname(r)))
    


#%% - Tau vs time / Temp
fig200, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_OH[:]):
        ax.plot(r.temperatura,r.tau,'.-',alpha=0.5)
ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('T (°C)')
plt.suptitle('OH en H$_2$O\ntau vs temperatura' )
plt.savefig('0_tau_vs_T_OH.png',dpi=300)
plt.show()


fig201, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_OH[:]):
        ax.plot(r.temperatura,r.tau,'.-',alpha=0.8)
ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('T (°C)')
ax.set_xlim(-5,5)
ax.set_ylim(200,360)
plt.suptitle('OH en H$_2$O\ntau vs temperatura' )
plt.savefig('0_tau_vs_T_OH_zoom.png',dpi=300)
#%% Templogs 

fig100, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_OH[:]):
        ax.plot(r.time,r.temperatura,'.-',alpha=0.5)
ax.grid()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.suptitle('OH en H$_2$O\ntemperatura vs time' )
plt.xlim(0,350)
plt.savefig('1_T_vs_time_OH.png',dpi=300)
plt.show()


#%%
T_all = np.concatenate([r.temperatura for r in res_OH])
Tmin, Tmax = T_all.min(), T_all.max()

cmap = mpl.colormaps['viridis']
norm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax)

fig201, ax = plt.subplots(figsize=(10,4), constrained_layout=True)

for r in res_OH:
    sc = ax.scatter(r.time, r.tau,c=r.temperatura,cmap=cmap, 
                    norm=norm,s=20)

ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('t (s)')
ax.set_xlim(0,350)
plt.suptitle('OH en H$_2$O\ntau vs time')

# --- Colorbar global ---
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('T (°C)')
plt.savefig('0_tau_vs_time_OH.png', dpi=300)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Ahora para CP en VS55
fig300, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_CP[:]):
        ax.plot(r.time,r.temperatura,'.-',alpha=0.5)
ax.grid()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.suptitle('CP en VS55\ntemperatura vs time' )
plt.xlim(0,350)
plt.savefig('1_T_vs_time_CP.png',dpi=300)
plt.show()
# %%
T_all = np.concatenate([r.temperatura for r in res_CP])
Tmin, Tmax = T_all.min(), T_all.max()

cmap = mpl.colormaps['viridis']
norm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax)

fig301, ax = plt.subplots(figsize=(10,4), constrained_layout=True)

for r in res_CP:
    sc = ax.scatter(r.time, r.tau,c=r.temperatura,cmap=cmap, 
                    norm=norm,s=20)

ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('t (s)')
ax.set_xlim(0,350)
plt.suptitle('CP en VS55\ntau vs time')

# --- Colorbar global ---
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('T (°C)')
plt.savefig('0_tau_vs_time_CP.png', dpi=300)    
#%%
fig400, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_CP[:]):
        ax.plot(r.temperatura,r.tau,'.-',alpha=0.5)
ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('T (°C)')
plt.suptitle('CP en VS55\ntau vs temperatura' )
plt.savefig('0_tau_vs_T_CP.png',dpi=300)
plt.show()


fig401, ax  = plt.subplots(figsize=(10,4),constrained_layout=True)
for i,r in enumerate(res_CP[:]):
        ax.plot(r.temperatura,r.tau,'.-',alpha=0.8)
ax.grid()
ax.set_ylabel('τ (ns)')
ax.set_xlabel('T (°C)')
ax.set_xlim(-11,-1)
ax.set_ylim(40,200)
plt.suptitle('CP en VS55\ntau vs temperatura' )
plt.savefig('0_tau_vs_T_CP_zoom.png',dpi=300)




#%%  Promedio los tau por temperatura
temp_OH1 = res_OH[0].temperatura 
tau_OH1 = res_OH[0].tau
temp_OH2 = res_OH[1].temperatura 
tau_OH2 = res_OH[1].tau
temp_OH3 = res_OH[2].temperatura 
tau_OH3 = res_OH[2].tau

#recorto a -20 - 20 °C
indx_1 = np.nonzero((temp_OH1>=-15)&(temp_OH1<=20))
indx_2 = np.nonzero((temp_OH2>=-15)&(temp_OH2<=20))
indx_3 = np.nonzero((temp_OH3>=-15)&(temp_OH3<=20))

temp_OH1=temp_OH1[indx_1] 
tau_OH1=tau_OH1[indx_1] 

temp_OH2=temp_OH2[indx_2] 
tau_OH2=tau_OH2[indx_2]

temp_OH3=temp_OH3[indx_3] 
tau_OH3=tau_OH3[indx_3]

# Concatenamos todas las temperaturas y taus
temp_OH_total = np.concatenate((temp_OH1, temp_OH2, temp_OH3))
tau_OH_total = np.concatenate((tau_OH1, tau_OH2, tau_OH3))

# Definimos los intervalos de temperatura
intervalo_temperatura = 1
temp_OH_intervalo = np.arange(np.min(temp_OH_total), np.max(temp_OH_total) + intervalo_temperatura, intervalo_temperatura)

# Lista para almacenar los promedios de tau
promedios_OH_tau = []
errores_OH_tau =[]

# Iteramos sobre los intervalos de temperatura
for temp in temp_OH_intervalo:
    # Seleccionamos los valores de tau correspondientes al intervalo de temperatura actual
    tau_intervalo = tau_OH_total[(temp_OH_total >= temp) & (temp_OH_total < temp + intervalo_temperatura)]
    
    # Calculamos el promedio y lo agregamos a la lista
    promedios_OH_tau.append(np.mean(tau_intervalo))
    errores_OH_tau.append(np.std(tau_intervalo))

# Convertimos la lista de promedios a un array de numpy
promedios_OH_tau = np.array(promedios_OH_tau)
err_temp_OH=np.full(len(temp_OH_intervalo),intervalo_temperatura/2)

print("Intervalo de Temperatura   |   Promedio de Tau    |")
print("-------------------------------------------------")
for i in range(len(temp_OH_intervalo)):
    print(f"{temp_OH_intervalo[i]:.2f} - {temp_OH_intervalo[i] + intervalo_temperatura:.2f} °C   |   {promedios_OH_tau[i]:.2e}")
#%% Ahora para CP en VS55
temp_CP1 = res_CP[0].temperatura 
tau_CP1 = res_CP[0].tau
temp_CP2 = res_CP[1].temperatura 
tau_CP2 = res_CP[1].tau
temp_CP3 = res_CP[2].temperatura 
tau_CP3 = res_CP[2].tau

#recorto a -20 - 20 °C
indx_1 = np.nonzero((temp_CP1>=-15)&(temp_CP1<=20))
indx_2 = np.nonzero((temp_CP2>=-15)&(temp_CP2<=20))
indx_3 = np.nonzero((temp_CP3>=-15)&(temp_CP3<=20))

temp_CP1=temp_CP1[indx_1] 
tau_CP1=tau_CP1[indx_1] 

temp_CP2=temp_CP2[indx_2] 
tau_CP2=tau_CP2[indx_2]

temp_CP3=temp_CP3[indx_3] 
tau_CP3=tau_CP3[indx_3]

# Concatenamos todas las temperaturas y taus
temp_CP_total = np.concatenate((temp_CP1, temp_CP2, temp_CP3))
tau_CP_total = np.concatenate((tau_CP1, tau_CP2, tau_CP3))

# Definimos los intervalos de temperatura
intervalo_temperatura = 1
temp_CP_intervalo = np.arange(np.min(temp_CP_total), np.max(temp_CP_total) + intervalo_temperatura, intervalo_temperatura)

# Lista para almacenar los promedios de tau
promedios_CP_tau = []
errores_CP_tau =[]

# Iteramos sobre los intervalos de temperatura
for temp in temp_CP_intervalo:
    # Seleccionamos los valores de tau correspondientes al intervalo de temperatura actual
    tau_intervalo = tau_CP_total[(temp_CP_total >= temp) & (temp_CP_total < temp + intervalo_temperatura)]
    
    # Calculamos el promedio y lo agregamos a la lista
    promedios_CP_tau.append(np.mean(tau_intervalo))
    errores_CP_tau.append(np.std(tau_intervalo))

# Convertimos la lista de promedios a un array de numpy
promedios_CP_tau = np.array(promedios_CP_tau)
err_temp_CP=np.full(len(temp_CP_intervalo),intervalo_temperatura/2)

print("Intervalo de Temperatura   |   Promedio de Tau    |")
print("-------------------------------------------------")
for i in range(len(temp_CP_intervalo)):
    print(f"{temp_CP_intervalo[i]:.2f} - {temp_CP_intervalo[i] + intervalo_temperatura:.2f} °C   |   {promedios_CP_tau[i]:.2e}")

#%% PLOT TAU SAR

fig, (ax,ax2) = plt.subplots(nrows=2,figsize=(10, 6),sharex=True, constrained_layout=True)
ax.set_title('OH en H$_2$O',loc='left')
ax.set_ylabel(r'$\tau$ (s)')
ax.errorbar(x=temp_OH_intervalo,y=promedios_OH_tau,xerr=err_temp_OH,yerr=errores_OH_tau,capsize=4,fmt='.-')
ax2.set_xlabel('Temperatura (°C)')

ax2.set_title('CP en VS55',loc='left')
ax2.set_ylabel(r'$\tau$ (s)')
ax2.errorbar(x=temp_CP_intervalo,y=promedios_CP_tau,xerr=err_temp_CP,yerr=errores_CP_tau,capsize=4,fmt='.-')

for a in [ax,ax2]:
    a.grid()
plt.xlim(-16,21)

plt.suptitle(r'$\tau$ vs temperatura',fontsize=14)
plt.savefig('tau_135_38_promedio.png',dpi=400)
plt.show()