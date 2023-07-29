import os
import re,sys
from pycaret.classification import setup,get_config,load_model,predict_model
import copy
import argparse
import zipfile
import redis
from sklearn.model_selection import LeaveOneOut
LOO = LeaveOneOut()
import MYSVG
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from AutoML import AutoML,train_models_with_ifs
from DataProcessing import *
from Bio.SeqIO import parse
from AODE import *
from zip import zip_folder
from collections import OrderedDict,Counter

redis_client = redis.Redis(host='localhost', port=6379, db=0)

TrSEQLENGTH=8
def readData(fastafile, pre,intype,test=False):
    global TrSEQLENGTH
    fasta_dict = OrderedDict()
    seqName = []
    if intype == 'fasta':
        if pre:
            if test:
                seqs = []
                labels=[]
                for record in parse(fastafile, "fasta"):
                    label = str(record.description).split()[-1]
                    if int(label) not in [0,1]:
                        print('Please the check sequence label!')
                        sys.exit(1)
                    seq = record.seq
                    if len(seq)!=TrSEQLENGTH:
                        print('Please check the sequence length of the test set, it does not match the sequence length of the training set!')
                        sys.exit(1)
                    seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                    if len(seq) == TrSEQLENGTH:
                        seqs.append(str(seq))
                        # seqName.append(str(record.id))
                        seqName.append(str(record.id) + '_' + str(int(TrSEQLENGTH/2)))
                        labels.append(int(label))
                fasta_dict['###'] = seqs
                return fasta_dict, seqName,labels
            else:
                
                seqs = []
                for record in parse(fastafile, "fasta"):
                    seq = record.seq
                    seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                    if len(seq) == TrSEQLENGTH:
                        seqs.append(str(seq))
                        seqName.append(str(record.id) + '_' + str(int(TrSEQLENGTH/2)))
                    elif len(seq) >= TrSEQLENGTH:
                        for j in range(0, len(seq) - TrSEQLENGTH + 1):
                            sub_seq = seq[j:j + TrSEQLENGTH]
                            seqs.append(str(sub_seq))
                            seqName.append(str(record.id) + '_' + str(j+int(TrSEQLENGTH/2)))
                    elif len(seq) in [8,10,12,14,16,18]:
                        sub_seq = '-' * int((TrSEQLENGTH - len(seq))/2) +seq + '-' * int((TrSEQLENGTH - len(seq))/2)
                        seqs.append(str(sub_seq))
                        seqName.append(str(record.id) + '_' + str(int(len(seq)/2)))
                    else:
                        sub_seq = seq + '-' * (TrSEQLENGTH - len(seq))
                        seqs.append(str(sub_seq))
                        seqName.append(str(record.id) + '_' + str(int(TrSEQLENGTH/2)))
                fasta_dict['###'] = seqs
                return fasta_dict, seqName
        else:
            checkPoint=0
            for record in parse(fastafile, "fasta"):
                seq = record.seq
                if len(seq) not in [8,10,12,14,16,18,20]:
                    print('Please check the sequence, it is not the required length!')
                    sys.exit(1)
                else:
                    if checkPoint==0:
                        TrSEQLENGTH=len(seq)
                        checkPoint=1
                if len(seq)!=TrSEQLENGTH:
                    print('Please check the sequence, it is not the same length!')
                    sys.exit(1)
                label = str(record.description).split()[-1]
                if int(label) not in [0,1]:
                    print('Please check the sequence label!')
                    sys.exit(1)
                seqName.append(str(record.id))
                
                seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                if label[-1] == '1':
                    fasta_dict[str(seq)] = '1'
                else:
                    fasta_dict[str(seq)] = '0'
            return fasta_dict, seqName
    else:
        if pre:
            if test:
                seqs = []
                labels=[]
                with open(fastafile, 'rt') as f:
                    content = f.readlines()
                    f.seek(0, 0)
                seqidx=1
                for eachline in content:
                    try:
                        eachline.split('\t')
                    except:
                        print('The format of input data is wrong, please check!')
                        sys.exit(1)
                    
                    label, seq = eachline.split('\t')[0], (eachline.split('\t')[1]).split('\n')[0]
                    if int(label) not in [0,1]:
                        print('Please check the sequence label!')
                        sys.exit(1)
                    if len(seq)!=TrSEQLENGTH:
                        print('Please check the sequence length of the test set, it does not match the sequence length of the training set!')
                        sys.exit(1)
                    if len(seq) == TrSEQLENGTH:
                        seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                        seqs.append(str(seq))
                        labels.append(str(label))
                    if str(label) == '1':
                        seqName.append('P'+str(seqidx))
                    else:
                        seqName.append('N'+str(seqidx))
                    seqidx+=1
                fasta_dict['###'] = seqs
                return fasta_dict, seqName,labels
            else:
                seqs = []
                with open(fastafile, 'rt') as f:
                    content = f.readlines()
                    f.seek(0, 0)
                seqidx=1
                for eachline in content:
                    eachline = (eachline.strip()).upper()
                    if len(eachline) == TrSEQLENGTH:
                        seq = eachline
                        seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                        seqs.append(str(seq))
                        seqName.append(str(int(seqidx)))
                    elif len(eachline) == 0:
                        continue
                    elif len(eachline) > TrSEQLENGTH:
                        for j in range(0, len(eachline) - TrSEQLENGTH + 1):
                            sub_seq = eachline[j:j + TrSEQLENGTH]
                            seqs.append(str(sub_seq))
                            seqName.append(str(seqidx)+'_'+str(j))
                    else:
                        seq = eachline + '-' * (TrSEQLENGTH - len(eachline))
                        seqs.append(str(seq))
                        seqName.append(str(int(seqidx)))
                    seqidx+=1
                fasta_dict['###'] = seqs
                return fasta_dict, seqName
        else:
            with open(fastafile, 'rt') as f:
                content = f.readlines()
                f.seek(0, 0)
            seqidx=1
            checkPoint=0
            for eachline in content:
                try:
                    eachline.split('\t')
                except:
                    print('The format of input data is wrong, please check!')
                    sys.exit(1)
                
                label, seq = eachline.split('\t')[0], (eachline.split('\t')[1]).split('\n')[0]
                if int(label) not in [0,1]:
                    print('Please check the sequence label!')
                    sys.exit(1)
                if len(seq) not in [8,10,12,14,16,18,20]:
                    # sys.stderr.write('>Please provide test sets!\n')
                    print('Please check the sequence, it is not the required length!')
                    sys.exit(1)
                else:
                    if checkPoint==0:
                        TrSEQLENGTH=len(seq)
                        checkPoint=1
                if len(seq)!=TrSEQLENGTH:
                    print('Please check the sequence, it is not the same length!')
                    sys.exit(1)
                seq = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(seq).upper())
                if str(label) == '1':
                    fasta_dict[str(seq)] = '1'
                    seqName.append('P'+str(seqidx))
                else:
                    fasta_dict[str(seq)] = '0'
                    seqName.append('N'+str(seqidx))
                seqidx+=1
            return fasta_dict, seqName

def read_config(file):
    parameters = {}
    with open(file) as f:
        records = f.readlines()
    for line in records:
        if line[0] == '#' or line.strip() == '':
            continue
        else:
            array = line.strip().split('=')
            parameters[array[0]] = array[1]

    for key in parameters:
        if parameters[key].isdigit():
            parameters[key] = int(parameters[key])

    default = {
        'ML': 'nb;svm;rf;lr;knn;xgboost;catboost;lightgbm;AODE',
        'Validation': 5
    }

    for key in default:
        if key in parameters:
            if parameters[key] == '':
                parameters[key] = default[key]
    return parameters
def load_matrix(proteasesOne):

    ff='finalModels/' + proteasesOne + '/allmatrix.json'
    cached_data = redis_client.get(str(proteasesOne)+'_matrix_json')
    if cached_data:

        return cached_data.decode('utf-8')
    else:

        with open(ff, 'r') as f:
            allmatrix_p = json.load(f) 
        json_allmatrix = json.dumps(allmatrix_p)

        redis_client.set(str(proteasesOne)+'_matrix_json', json_allmatrix)
        redis_client.expire(str(proteasesOne)+'_matrix_json',3600) 

    return json_allmatrix


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(usage="it's usage tip.")
    parser.add_argument('--inputType', default='fasta',type=str, help='fasta or peptide.')
    parser.add_argument("--config", default='config.txt', help="The path to the config file.")
    parser.add_argument('--trainfile', default=None, type=str, help='The path to the training set file containing the sequences in fasta(peptide) format, where the length of the sequences is 8, 10, 12, 14, 16, 18 or 20.')
    parser.add_argument('--protease', type=str, help='The protease you want to predict cleavage to, eg: A01.001' + '\n'
                        'Or if you want to build a new model, please create a name. There should no space in the model name.')
    parser.add_argument('--outputpath', type=str, help='The path of output.')
    parser.add_argument('--testfile', default=None,type=str, help='The path to the test set file containing the sequences in fasta(peptide) format, where the length of the sequences is 8, 10, 12, 14, 16, 18 or 20. If not, it will be divided from the training set.')
    parser.add_argument('--predictfile', default=None, type=str, help='The path to the prediction data file containing the protease sequences in fasta(peptide) format, where the length of the sequences is 8/10/12/14/16/18/20.')
    parser.add_argument('--mode', type=str, required=True ,help='Choose the mode, three modes can be used: prediction, TrainYourModel, UseYourOwnModel, only select one mode each time.')
    parser.add_argument('--modelfile',default=None, type=str, help='The path to the trained model generated from the TrainYourModel module. eg 0_model' ) 
    parser.add_argument('--SHAP', default=None, type=str, help='Select Yes or No to control the program to calculate SHAP.')
    parser.add_argument('--PLOT', default=None, type=str, help='Select Yes or No to control whether the program computes the visualization of cleavage sites.')
    parser.add_argument('--processNum', default=2, type=int, help='The number of processes in the program. Note: Integer values represent the number of processes. --processNum setting can speed up the running efficiency of the program, but it also takes up more computing resources.')
    args = parser.parse_args()

    # fastafile=args.trainfile
    # testFastaFile=args.testfile
    # predictFastaFile=args.predictfile
    # templatefolder = args.outputpath

    if args.mode =='TrainYourModel':
        # from pyspark.sql import SparkSession
        # spark = SparkSession.builder.getOrCreate()
        # spark.sparkContext.setLogLevel("WARN")
        # from pycaret.parallel import FugueBackend
            
        templatefolder = args.outputpath
        
        parameters=read_config(args.config)

        # check input
        fastafile=args.trainfile
        testFastaFile=args.testfile
        predictFastaFile=args.predictfile

        if len(parameters['ML'])==0:
            print('Please select at least one model!')
            sys.exit(1)
        allmatrixName=args.outputpath+'/'+'allmatrix.json'
        
        if args.trainfile == None:
            print('Please provide train sets!')
            sys.exit(1)
        
        if args.inputType == 'fasta':
            fasta_dict, train_seqname = readData(args.trainfile, False, args.inputType,test=False)  
            if args.testfile:
                testfasta_dict, test_seqname,test_labels = readData(args.testfile, True, args.inputType,test=True)  
            if args.predictfile:
                predictfasta_dict, predict_seqname = readData(args.predictfile, True, args.inputType,test=False)
        else:
            fasta_dict,pre_pos = readData(args.trainfile, False,args.inputType,test=False)
            if args.testfile:
                testfasta_dict,test_pre_pos,test_labels  = readData(args.testfile, True, args.inputType,test=True)
            if args.predictfile:
                predictfasta_dict, predict_seqname = readData(args.predictfile, True, args.inputType,test=False)
                
         
        if args.predictfile:
            if args.testfile:
                data,testdata,predata=Dataprocessing(args,fasta_dict,testfasta_dict,test_labels,predictfasta_dict,allmatrixName,TrSEQLENGTH)
            else:
                data,testdata,predata=Dataprocessing(args,fasta_dict,fasta_dict,None,predictfasta_dict,allmatrixName,TrSEQLENGTH)
        else:
            if args.testfile:
                data,testdata=Dataprocessing(args,fasta_dict,testfasta_dict,test_labels,None,allmatrixName,TrSEQLENGTH)
            else:
                data,testdata=Dataprocessing(args,fasta_dict,fasta_dict,None,None,allmatrixName,TrSEQLENGTH)

        # set environment        
        numeric_features = []
        for col in data.columns:
            if 'KNN' in col:
                numeric_features.append(col)
        if parameters['Validation'] == 'LOO':

            exp = setup(data=data, target='true_label', test_data=testdata, session_id=123, use_gpu=False,
                        normalize=False, remove_perfect_collinearity=False,
                        fold_strategy=LOO, silent=True, verbose=False, feature_selection=False, preprocess=True,
                        numeric_features=numeric_features) 
        else: 
            exp = setup(data=data, target='true_label', test_data=testdata, session_id=123, use_gpu=False,
                        normalize=False, remove_perfect_collinearity=False,
                        fold=parameters['Validation'], silent=True, verbose=False, feature_selection=False,preprocess=True,
                        numeric_features=numeric_features)
        

        x_train = get_config('X_train')
        y_train = get_config('y_train')
        x_test = get_config('X_test')
        y_test = get_config('y_test')
        MLs = parameters['ML'].split(';') 
        selectedModels = copy.deepcopy(MLs)
        if 'AODE' in selectedModels:
            del selectedModels[selectedModels.index('AODE')]
        fold_CV=parameters['Validation']
        
        if len(selectedModels) >= 3 :
            n_select = 3  

            if args.predictfile:
                AutoML(args,selectedModels,n_select,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,'all',MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH)
            else:
                AutoML(args,selectedModels,n_select,templatefolder,x_train,y_train,x_test,y_test,None,fold_CV,'all',MLs,allmatrixName,None,None,TrSEQLENGTH)
        elif len(selectedModels)==2: 

            if args.predictfile:
                train_models_with_ifs(args,selectedModels,data,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,'all',MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH)
            else:
                
                train_models_with_ifs(args,selectedModels,data,templatefolder,x_train,y_train,x_test,y_test,None,fold_CV,'all',MLs,allmatrixName,None,None,TrSEQLENGTH)
        
        elif len(selectedModels)==1:
            if args.predictfile:
                train_models_with_ifs(args,selectedModels,data,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,'all',MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH)
                
            else:
                train_models_with_ifs(args,selectedModels,data,templatefolder,x_train,y_train,x_test,y_test,None,fold_CV,'all',MLs,allmatrixName,None,None,TrSEQLENGTH)
        elif len(selectedModels)==0:

            if args.predictfile:
                train_models_with_ifs(args,None,data,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,'all',MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH)
            else:
                train_models_with_ifs(args,None,data,templatefolder,x_train,y_train,x_test,y_test,None,fold_CV,'all',MLs,allmatrixName,None,None,TrSEQLENGTH)
    elif args.mode=='prediction':
        templatefolder = args.outputpath
        predictFastaFile=args.predictfile
        TrSEQLENGTH1=8
        
        if args.predictfile==None:  
            print('Please provide prediction sets!')
            sys.exit(1)
        with open(args.predictfile) as f:
            content = f.read()
        ###check fasta format
        if '>' not in content:
            print('The input file seems not in fasta format.')
            sys.exit(1)
        
        proteaeList = str(args.protease).strip().split(',')
        if len(proteaeList)==0:
            print('Please select at least one protease!')
            sys.exit(1)
        
        if args.inputType == 'fasta':
            fasta_dict, seqname = readData(predictFastaFile, True, args.inputType,test=False) 
            # print(fasta_dict,seqname)
        else:
            print('The input file seems not in fasta format.')
            sys.exit(1)

        outF='results'
        os.mkdir('%s/%s' % (templatefolder, outF))
        checkpoint_pre = 0

        for proteasesOne in proteaeList:
            # print(proteasesOne)
            resultDF = pd.DataFrame()
            resultDF['seqs'] = fasta_dict['###']
            # allmatrix=load_matrix(proteasesOne) # 
            # allmatrix=json.loads(allmatrix)
            ff='finalModels/' + proteasesOne + '/allmatrix.json'
            with open(ff, 'r') as f:
                allmatrix = json.load(f) 
            
            predata=Pre_processing(args,fasta_dict,allmatrix,proteasesOne,TrSEQLENGTH1)

            # AODE
            if proteasesOne in ['M24.026','S01.140','M10.014','M12.225','S01.161','S01.302','S08.109','M12.005','M16.003']:
                aodemodelfile='finalModels/' + proteasesOne + '/' + proteasesOne + '_AODE_model'
                FEATURES=np.load('finalModels/' + proteasesOne +'/featuresSorted.npy')
                predata_aode=predata.loc[:, FEATURES]
                AODE_predlabels,AODE_predscores=AODE_predict(args,aodemodelfile,predata_aode)
                if args.inputType == 'fasta':
                    seqN=['_'.join(v.split('_')[0:-1]) for v in seqname]
                    CleavageP=[v.split('_')[-1] for v in seqname]
                    resultDF.insert(0, 'sequence_id', seqN)
                    resultDF.insert(1, 'position', CleavageP)
                resultDF.insert(0, 'protease', [proteasesOne] * len(fasta_dict['###']))
                resultDF['prediction'] = AODE_predlabels
                resultDF['pro'] = AODE_predscores

            else:
                
                loadedModel = 'finalModels/' + proteasesOne + '/MLmodel'  
                FEATURES=np.load('finalModels/' + proteasesOne +'/featuresSorted.npy')
                # print(FEATURES)
                predata2=predata.loc[:, FEATURES]
                clf = load_model(loadedModel)
                preds = predict_model(clf, data=predata2,raw_score=True)
                if args.inputType == 'fasta':

                    seqN=['_'.join(v.split('_')[0:-1]) for v in seqname]
                    CleavageP=[v.split('_')[-1] for v in seqname]
                    resultDF.insert(0, 'sequence_id', seqN)
                    resultDF.insert(1, 'position', CleavageP)

                resultDF.insert(0, 'protease', [proteasesOne] * len(fasta_dict['###']))
                resultDF['prediction'] = list(preds['Label'].values)
                
                if 'Score_1' in preds.columns:
                    Pro_List=[round(float(v),4) for v in list(preds['Score_1'].values)]
                    resultDF['pro'] = Pro_List
                elif 'Score' in preds.columns:

                    Pro_List=[]
                    for i,pr in enumerate(list(preds['Label'].values)):
                        if int(pr) ==1:
                            Pro_List(round(float(list(preds['Score'].values)[i]),3))
                        elif int(pr) ==0:
                            Pro_List.append(round(float(1-list(preds['Score'].values)[i]),3))
                    resultDF['pro'] = Pro_List
                else:
                    resultDF['pro'] = list(preds['Label'].values) #

            if checkpoint_pre == 0:
                AllresultDF = resultDF
                checkpoint_pre = 1
            else:
                AllresultDF = AllresultDF.append(resultDF)

        AllresultDF.to_csv(args.outputpath + '/results.csv', index=False)
        Count=Counter(resultDF['sequence_id'].values)

        with open(args.predictfile, "r") as handle:
            records = list(parse(handle, "fasta"))
        for i,sid in enumerate(list(Count.keys())):
            thirdFN=str(i)+'_'+sid
            os.mkdir('%s/%s/%s' % (templatefolder,outF,thirdFN))
            with open(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','w') as f_tmp:
                f_tmp.write('>'+str(records[i].id)+'\n')
                f_tmp.write(str(records[i].seq)+'\n')
                
            result_tmp=AllresultDF.loc[AllresultDF['sequence_id']==str(sid)]
            rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
            result_tmp.insert(0, 'rank', rANK)
            result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
            # plot
            if args.PLOT=='Yes':
                os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot1'))
                os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot2'))
                for pta in proteaeList:
                    rTmp= result_tmp.loc[(result_tmp['protease']==pta)]
                    rTmp.to_csv('{}/{}/{}/{}_{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN,pta), index=False)
                    svg_outFile='{}/{}/{}/{}_plot1/{}_{}.svg'.format(templatefolder, outF,thirdFN,thirdFN,thirdFN,pta)
                    rTmp1= rTmp.loc[(rTmp['protease']==pta) & (rTmp['prediction']==1)]
                    try:
                        MYSVG.createBarChart(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','HIGH',rTmp1,svg_outFile)
                        SVGyes='Yes'    
                    except:
                        SVGyes='No'
                    if SVGyes=='Yes':
                        zip_folder(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1',templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1'+'.zip')
                    # R
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
    elif args.mode=='UseYourOwnModel':
        templatefolder = args.outputpath
        predictFastaFile=args.predictfile
        if args.predictfile==None:
            print('Please provide prediction sets!')
            sys.exit(1)
        with open(args.predictfile) as f:
            content = f.read()
        ###check fasta format
        if '>' not in content:
            print('The input file seems not in fasta format.')
            sys.exit(1)
            
        os.mkdir('%s/%s' % (templatefolder, 'model_files'))
        modelfiles = templatefolder+'/'+'model_files'
        outF='results'
        os.mkdir('%s/%s' % (templatefolder, outF))
        
        upload_file=args.modelfile
        Unzip =zipfile.ZipFile(upload_file, 'r')
        files_name= Unzip.namelist()
        model_type=''
        error_file=''
        for file_z in files_name:
            if 'AODE' in file_z:
                model_type='type'
                modelfilename=file_z
            elif 'MLmodel.pkl' in file_z:
                modelfilename=file_z
            elif 'allmatrix.json' in file_z:
                matrixname=file_z
            elif 'featuresSorted' in file_z:
                features_sorted=file_z
            elif 'TrSEQLENGTH' in file_z:
                TrSEQLENGTH_path=file_z
            else:
                error_file='Yes'
        if error_file=='Yes':
            print('Please check your model!')
            sys.exit(1)
            
        for zfile in files_name:
            Unzip.extract(zfile,modelfiles)

        TrSEQLENGTH=int(np.load(modelfiles +'/'+TrSEQLENGTH_path))
        # print(TrSEQLENGTH)
        
        fasta_dict, seqname = readData(predictFastaFile, True, args.inputType,test=False)
        resultDF = pd.DataFrame()
        resultDF['seqs'] = fasta_dict['###']
        
        with open(modelfiles + '/'+matrixname, 'r') as f:
            allmatrix = json.load(f)
        # data prcessing    
        predata=Pre_processing(args,fasta_dict,allmatrix,args.protease,TrSEQLENGTH)
        FEATURES=np.load(modelfiles +'/'+features_sorted)
        predata=predata.loc[:, FEATURES]
        # model
        _modelfile=modelfiles +'/'+modelfilename
        if model_type =='type':
                
            currentpath = os.getcwd()
            AODE_predlabels,AODE_predscores=AODE_predict(args,_modelfile,predata)
            if args.inputType == 'fasta':
                seqN=['_'.join(v.split('_')[0:-1]) for v in seqname]
                CleavageP=[v.split('_')[-1] for v in seqname]
                resultDF.insert(0, 'sequence_id', seqN)
                resultDF.insert(1, 'position', CleavageP)
            #resultDF.insert(0, 'protease', [args.protease] * len(fasta_dict['###']))
            resultDF['prediction'] = AODE_predlabels
            resultDF['pro'] = AODE_predscores
        else: 
            if '.pkl' in _modelfile:
                _modelfile=_modelfile[:-4]
            else:
                print('Please check your upload files!')
                sys.exit(1)
            print(_modelfile)
            clf = load_model(_modelfile)

            preds = predict_model(clf, data=predata,raw_score=True)
            if args.inputType == 'fasta':
                seqN=['_'.join(v.split('_')[0:-1]) for v in seqname]
                CleavageP=[v.split('_')[-1] for v in seqname]
                resultDF.insert(0, 'sequence_id', seqN)
                resultDF.insert(1, 'position', CleavageP)

            resultDF.insert(0, 'protease', [args.protease] * len(fasta_dict['###']))
            resultDF['prediction'] = list(preds['Label'].values)
            
            if 'Score_1' in preds.columns:
                
                Pro_List=[round(float(v),4) for v in list(preds['Score_1'].values)]
                resultDF['pro'] = Pro_List
            elif 'Score' in preds.columns:
                Pro_List=[]
                for i,pr in enumerate(list(preds['Label'].values)):
                    if int(pr) ==1:
                        Pro_List.append(round(float(list(preds['Score'].values)[i]),3))
                    elif int(pr) ==0:
                        Pro_List.append(round(float(1-list(preds['Score'].values)[i]),3))
                resultDF['pro'] = Pro_List
            else:
                resultDF['pro'] = list(preds['Label'].values) 

        resultDF.to_csv(args.outputpath + '/results.txt', index=False)

        Count=Counter(resultDF['sequence_id'].values)
        with open(args.predictfile, "r") as handle:
            records = list(parse(handle, "fasta"))
            
        for i,sid in enumerate(list(Count.keys())):
            thirdFN=str(i)+'_'+sid
            os.mkdir('%s/%s/%s' % (templatefolder, outF,thirdFN))
            with open(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','w') as f_tmp:
                f_tmp.write('>'+str(records[i].id)+'\n')
                f_tmp.write(str(records[i].seq)+'\n')
            result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
            rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
            result_tmp.insert(0, 'rank', rANK)
            result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
            # plot
            if args.PLOT=='Yes':
                rTmp= result_tmp
                os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot1'))
                svg_outFile='{}/{}/{}/{}_plot1/{}.svg'.format(templatefolder, outF,thirdFN,thirdFN,thirdFN)
                rTmp1= rTmp.loc[(rTmp['prediction']==1)]
                try:
                    MYSVG.createBarChart(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','HIGH',rTmp1,svg_outFile)
                    SVGyes='Yes'
                except:
                    SVGyes='No'
                if SVGyes=='Yes':
                    zip_folder(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1',templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot1'+'.zip')
                # R
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
                re_r2.to_csv('{}/{}/{}/{}.txt'.format(templatefolder, outF,thirdFN,thirdFN),index=False,header=None)

                os.mkdir('%s/%s/%s/%s' % (templatefolder, outF,thirdFN,thirdFN+'_plot2'))
                inPath2R1='{}/{}/{}/{}_plot2/{}'.format(templatefolder, outF,thirdFN,thirdFN,thirdFN)
                inPath2R2='{}/{}/{}/{}'.format(templatefolder, outF,thirdFN,thirdFN)
                Rstr='Rscript visual.R '+sid+' '+inPath2R1+' '+inPath2R2 # change path
                try:
                    subprocess.run(Rstr, shell=True)
                    Ryes='Yes'
                except:
                    Ryes='No'
                if Ryes=='Yes':
                    zip_folder(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot2',templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'_plot2'+'.zip')
    else:
        print('Please check the value of "mode"!')
