
import numpy as np


def Concat_epochs(DB):
    A = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_40percent_epoch100.npy")
    B = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_50percent_epoch100.npy")
    C = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_60percent_epoch100.npy")
    D = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_70percent_epoch100.npy")
    E = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_80percent_epoch100.npy")
    F = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_90percent_epoch100.npy")


    AA = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_40percent_epoch200.npy")
    BB = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_50percent_epoch200.npy")
    CC = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_60percent_epoch200.npy")
    DD = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_70percent_epoch200.npy")
    EE = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_80percent_epoch200.npy")
    FF = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_90percent_epoch200.npy")


    AAA = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_40percent_epoch300.npy")
    BBB = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_50percent_epoch300.npy")
    CCC = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_60percent_epoch300.npy")
    DDD = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_70percent_epoch300.npy")
    EEE = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_80percent_epoch300.npy")
    FFF = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_90percent_epoch300.npy")

    AAAA = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_40percent_epoch400.npy")
    BBBB = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_50percent_epoch400.npy")
    CCCC = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_60percent_epoch400.npy")
    DDDD = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_70percent_epoch400.npy")
    EEEE = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_80percent_epoch400.npy")
    FFFF = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_90percent_epoch400.npy")

    AAAAA = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_40percent_epoch500.npy")
    BBBBB = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_50percent_epoch500.npy")
    CCCCC = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_60percent_epoch500.npy")
    DDDDD = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_70percent_epoch500.npy")
    EEEEE = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_80percent_epoch500.npy")
    FFFFF = np.load(f"Analysis/Performance_Analysis/{DB}/metrics_90percent_epoch500.npy")


    A1=np.stack([A,B,C,D,E,F],axis=1)
    A2=np.stack([AA,BB,CC,DD,EE,FF],axis=1)
    A3=np.stack([AAA,BBB,CCC,DDD,EEE,FFF],axis=1)
    A4=np.stack([AAAA,BBBB,CCCC,DDDD,EEEE,FFFF],axis=1)
    A5=np.stack([AAAAA,BBBBB,CCCCC,DDDDD,EEEEE,FFFFF],axis=1)

    np.save(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs100.npy",A1)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs200.npy", A2)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs300.npy", A3)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs400.npy", A4)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs500.npy", A5)

    # A11=np.load(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_100.npy")
    # A22=np.load(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_200.npy")
    # A33=np.load(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_300.npy")
    # A44=np.load(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_400.npy")
    # A55=np.load(f"Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_500.npy")
    #
    #
    # return A11,A22,A33,A44,A55
