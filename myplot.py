import os
import matplotlib.pyplot as plt
import matplotlib
import shap
import pandas as pd
import MYSVG
import subprocess
from Bio.SeqIO import parse
from itertools import cycle
from pycaret.classification import interpret_model
from zip import zip_folder

def SHAP_plot(args,templatefolder,MoldeNames,Models,x_train,x_test):
    savepath=templatefolder
    r=0
    if 'bagging' not in MoldeNames[r] and 'Stacking' not in MoldeNames[r] and 'Voting' not in MoldeNames[r]:
        if MoldeNames[r] in ['et','xgboost','catboost','lightgbm','rf','dt']: 
            print('*****',MoldeNames[r],'SHAP********')
            try:
                if args.testfile:
                    interpret_model(Models[r],plot='summary',save=savepath) #dt, et, catboost, rf, xgboost, lightgbm
                else:
                    interpret_model(Models[r],plot='summary',save=savepath,use_train_data=True)
                os.rename(savepath+'/'+'SHAP summary.png', savepath+'/'+'_'.join(MoldeNames[r].split())+'_SHAP.png')
            except:
                print('*****',MoldeNames[r],'An error occurred while calculating SHAP!********')
                
        else:
            print('*****',MoldeNames[r],'SHAP********')
            try:
                kmeansData=shap.kmeans(x_train,10)
                # sampleData=shap.sample(x_test,10)
                explainer1 = shap.KernelExplainer(Models[r].predict_proba,kmeansData, link="logit")
                if args.testfile:
                    shap_values1 = explainer1.shap_values(x_test, nsamples=100)
                    shap.summary_plot(shap_values1[1],x_test,show=False,save=True,path=savepath+'/'+'_'.join(MoldeNames[r].split())+'_SHAP.png')
                else:
                    shap_values1 = explainer1.shap_values(x_train, nsamples=100)
                    shap.summary_plot(shap_values1[1],x_train,show=False,save=True,path=savepath+'/'+'_'.join(MoldeNames[r].split())+'_SHAP.png') # Need to change source code
            except:
                print('*****',MoldeNames[r],'An error occurred while calculating SHAP!********')
    elif 'bagging'  in MoldeNames[r] or 'Stacking'  in MoldeNames[r] or 'Voting'  in MoldeNames[r]:
        print('*****',MoldeNames[r],'SHAP********')
        try:
            kmeansData=shap.kmeans(x_train,10)
            explainer1 = shap.KernelExplainer(Models[r].predict_proba,kmeansData, link="logit")
            if args.testfile:
                shap_values1 = explainer1.shap_values(x_test, nsamples=100)
                shap.summary_plot(shap_values1[1],x_test,show=False,save=True,path=savepath+'/'+'_'.join(MoldeNames[r].split())+'_SHAP.png')
            else:
                shap_values1 = explainer1.shap_values(x_train, nsamples=100)
                shap.summary_plot(shap_values1[1],x_train,show=False,save=True,path=savepath+'/'+'_'.join(MoldeNames[r].split())+'_SHAP.png')
        except:
            print('*****',MoldeNames[r],'An error occurred while calculating SHAP!********')
    
def plotAuc(fpr, tpr, auc_score, labelTag, isShow, path):
    colorstyle = ['b--', 'g--', 'r--']
    colors = cycle(['darkorange', 'cornflowerblue', 'deeppink', 'navy', 'aqua'])
    dpi = 300
    xylabel_fontsize = 15
    # linestyle1 = ['-.',':','.']
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
   

    for i, color in zip(range(len(labelTag)), colors):
        plt.plot(fpr[i], tpr[i], color=color, linestyle='-.',
                 lw=1, label=labelTag[i] + ' Auc = %0.3f' % auc_score[i])  #
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    plt.grid(True, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc='lower right', edgecolor='k', fancybox=False)
    plt.title('ROC of top3 classifiers')
    plt.xlabel('False Positive Rate', fontsize=xylabel_fontsize)
    plt.ylabel('True Positive Rate', fontsize=xylabel_fontsize)

    plt.tight_layout()
    plt.savefig(path + '/' + 'Auc.png', dpi=dpi)
    if isShow:
        plt.show()
    plt.close()

def visual_plot(templatefolder,pta,outF,thirdFN,result_tmp,sid):
    os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot1'))
    os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot2'))

    rTmp= result_tmp
    rTmp.to_csv('{}/{}/{}/{}_{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN,pta), index=False)
    svg_outFile='{}/{}/{}/{}_plot1/{}_{}.svg'.format(templatefolder, outF,thirdFN,thirdFN,thirdFN,pta)
    rTmp1= rTmp.loc[(rTmp['prediction']==1)]
    try:
        MYSVG.createBarChart(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','HIGH',rTmp1,svg_outFile)
        SVGyes='Yes'    
    except:
        SVGyes='No'
    if SVGyes=='Yes':
        zip_folder(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1',templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1'+'.zip')
    p_score={}

    for i,v in enumerate(list(rTmp['position'].values)):
        p_score[int(v)]=list(rTmp['pro'].values)[i]
        
    with open(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta', "r") as handle:
        record_r = list(parse(handle, "fasta"))
    recrd_seq=str(record_r[0].seq)
    position_score=[]

    for i in range(1,len(recrd_seq)+1):
        if i not in list(rTmp['position'].apply(int).values):
            position_score.append(float(0))
        else:
            position_score.append(float(p_score[i]))

    re_r2=pd.DataFrame(columns=['position','score','AA'])
    re_r2['position']=[int(i+1) for i in range(len(recrd_seq))]
    re_r2['score']=position_score
    re_r2['AA']=[recrd_seq[i] for i in range(len(recrd_seq))]
    re_r2.to_csv('{}/{}/{}/{}_{}.txt'.format(templatefolder, outF,thirdFN,thirdFN,pta),index=False,header=None)
    
    inPath2R1='{}/{}/{}/{}_plot2/{}_{}'.format(templatefolder, outF,thirdFN,thirdFN,thirdFN,pta)
    inPath2R2='{}/{}/{}/{}_{}'.format(templatefolder, outF,thirdFN,thirdFN,pta)
    Rstr='Rscript visual.R '+sid+' '+inPath2R1+' '+inPath2R2 # change path
    # print(Rstr)
    try:
        # os.system(Rstr)
        subprocess.run(Rstr, shell=True)
        Ryes='Yes'
    except:
        Ryes='No'
    if Ryes=='Yes':
        zip_folder(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot2',templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot2'+'.zip')
