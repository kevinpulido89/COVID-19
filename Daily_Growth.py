#COVID-19
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx')

LAT = pd.DataFrame()

paises_suramericanos = ['Mexico', 'Guatemala', 'Honduras', 'El_Salvador', 'Nicaragua',
                        'Costa_Rica', 'Panama', 'Colombia', 'Venezuela', 'Ecuador',
                        'Peru', 'Bolivia', 'Chile', 'Paraguay','Brazil',
                        'Argentina', 'Uruguay', 'Cuba', 'Dominican_Republic', 'Guyana',
                         'Suriname','Haiti']

for p in paises_suramericanos:
    tmp = df[df['countriesAndTerritories']==p]['cases'].sum()
    tmp2 = df[df['countriesAndTerritories']==p]['deaths'].sum()
    LAT = LAT.append({'Country' : p ,
                    'Total Cases' : int(tmp),
                    'Total Deaths': int(tmp2)},

                   ignore_index=True)

LAT.sort_values(by='Total Cases', ascending=False, axis=0, inplace=True)
LAT.set_index('Country', inplace=True)
print(LAT)

axes = LAT.plot.bar(rot=90, subplots=True)
plt.tight_layout()
plt.show()

COL = df[df['countriesAndTerritories']=='Colombia']

def cases_counter(df):
    cs = list(df['cases'])[::-1]
    return cs

####################---> FUNCIONES <---##########################
def acc_list(old):
    new = list()
    acc = 0
    for i in old:
        acc +=i
        new.append(acc)
    return new

def cases_counter(df):
    cs = list(df['cases'])[::-1]
    return cs

def PlotPolly(model, independent_variable, dependent_variabble, N):
    x_new = np.linspace(0, N, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Crecimiento de casos de COVID-19 en Colombia.')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel('Días')
    plt.ylabel('Cantidad de casos')
    plt.legend('Casos en Colombia','modelo')
    plt.show()
    # plt.close()
##################---> FIN FUNCIONES <---########################

plt.plot(cases_counter(COL))
plt.xlabel("Días")
plt.show()

acc = acc_list(cases_counter(COL))
yy = [i for i in range(len(acc))]

acc_col = pd.DataFrame()
window_size=3
for i in range(len(acc)-window_size):
    K = np.array(acc[i:i+window_size+1])
    acc_col = acc_col.append(pd.DataFrame(K).T)

# Here we use a polynomial of the 6th order
order = 5
f = np.polyfit(yy, acc, order)
# print('f: ',f)
p = np.poly1d(f)

print('El modelo ajustado es: {}'.format(p))

PlotPolly(p, yy, acc, len(acc))

# Prediccion para el siguente día
print(f'Para el día {len(acc)} se proyectan {p(len(acc))} casos')
