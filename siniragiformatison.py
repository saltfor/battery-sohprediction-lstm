
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
data = pd.read_csv("B05_discharge_soh.csv")

yenisatirvoltage= pd.DataFrame()
yenisatircurrent= pd.DataFrame()
yenisatirtemperature= pd.DataFrame()
yenisatirsoh= pd.DataFrame()
birlestirilendizi=pd.DataFrame(columns=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
                                        'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10',
                                        't1','t2','t3','t4','t5','t6','t7','t8','t9','t10','soh1'])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))

volt=pd.DataFrame()
curr=pd.DataFrame()
temp=pd.DataFrame()


cycle=data.iloc[:1,7:8]
soh=data.iloc[:1,8:9]
cycledegeri=int(cycle.loc[0])
cyclesatiradet=0
eskicyclesatiradet=0
j=0
while j<50285:
    
    for k in range(0,500):
        konum=data.iloc[j+k:j+k+1,7:8]
        if(j+k<50285):
            if(cycledegeri==int(konum.loc[j+k])):
                continue
            else:
                cycledegeri=int(konum.loc[j+k])
                cyclesatiradet=k+1
                break
        else:
            cyclesatiradet=k
            break
            
    
    soh=data.iloc[j:j+1,8:9]
    data1=data.iloc[j:j+cyclesatiradet+1,0:3]
    data2=data.iloc[j:j+cyclesatiradet+1,7:9]
    
    
    
    scaledvoltage = scaler.fit_transform(data1.iloc[:,0:1])
    scaledcurrent = scaler2.fit_transform(data1.iloc[:,1:2])
    scaledtemp = scaler3.fit_transform(data1.iloc[:,2:3])
    volt=pd.DataFrame(scaledvoltage)
    curr=pd.DataFrame(scaledcurrent)
    temp=pd.DataFrame(scaledtemp)   
    
    
    for i in range (0,cyclesatiradet):
        if(i<cyclesatiradet-10):
            a=0+i
            b=a+10
            yenisatirvoltage=volt.iloc[a:b,0:1]
            yenisatircurrent=curr.iloc[a:b,0:1]
            yenisatirtemperature=temp.iloc[a:b,0:1]
            
            yenisatirvoltage.columns=['satir']
            yenisatircurrent.columns=['satir']
            yenisatirtemperature.columns=['satir']
        
            yenisatirvoltage=yenisatirvoltage.transpose()
            yenisatircurrent=yenisatircurrent.transpose()
            yenisatirtemperature=yenisatirtemperature.transpose()
        
            yenisatirvoltage.columns=['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']
            yenisatircurrent.columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
            yenisatirtemperature.columns=['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']
        
            yenisatirsoh=soh.iloc[0:1,0:1]
            yenisatirsoh.columns=['soh1']
            yenisatirsoh=yenisatirsoh.transpose()
            yenisatirsoh.columns=['satir']
            yenisatirsoh=yenisatirsoh.transpose()
        
            satirlar=pd.concat((yenisatirvoltage,yenisatircurrent),axis=1)
            satirlar2=pd.concat((yenisatirtemperature,yenisatirsoh),axis=1)
            sonsatir=pd.concat((satirlar,satirlar2),axis=1)
            birlestirilendizi.loc[j+i]=sonsatir.loc['satir']
            
    j=j+cyclesatiradet
    
compression_opts = dict(method='zip',archive_name='B05_birlestirilmis.csv')  
birlestirilendizi.to_csv('B05_birlestirilmis.zip', index=False, compression=compression_opts) 
