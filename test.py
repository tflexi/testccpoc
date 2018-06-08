
import math
import glob

import pandas as pd
import numpy as np
import json
import time
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from flask import Flask
from flask import request



# In[]: flask + expense module


app = Flask(__name__)

@app.route('/expensePOC', methods = ['POST'])

def expenseModel_CC(cntLimIntRate = 3.00
                    ,cntBalMonths = 36
                    ,obsRecentRepayTxnBar = 31
                    ,syph = 2
                    ,infRecentTxnBar = 31
                    ,infCountTxnBar = 3
                    ,fuzzThreshold = 85                
                    ):

    jSonData = request.json

    #grab app context
    appRef = jSonData['reference']
    appDate = jSonData['submissionTime'][:10]    
    
    #stab commitment tables - observed & inferred
    obsCcVar = 'cntType accType accName accNum accBsb balCurrOut balAvail limit intRate minAmtDue minAmtDueDate cntLim cntBal cntObs cntRepayRatioObs'.split()
    ccCntObs = pd.DataFrame([[0]*len(obsCcVar)], columns=obsCcVar)  
    
    infCcVar = 'cntType text infNarration txtHit amount cntInf accType accName accNum accBsb'.split()
    ccCntInf = pd.DataFrame([[0]*len(infCcVar)],columns=infCcVar)    

    #cycle through accounts on bankstatment;
    for accIdx in np.arange(len(jSonData['bankData']['bankAccounts'])):
        
        #observed CC - get account characteristics - take conservative positions where no data        
        if jSonData['bankData']['bankAccounts'][accIdx]['accountType'].upper().replace(" ", "") == 'CREDITCARD':
            cntType = 'obs'
            accName = jSonData['bankData']['bankAccounts'][accIdx]['accountName']
            accType = jSonData['bankData']['bankAccounts'][accIdx]['accountType']
            accNum = jSonData['bankData']['bankAccounts'][accIdx]['accountNumber']
            accBsb = jSonData['bankData']['bankAccounts'][accIdx]['bsb']
            balCurrOut = abs(float(jSonData['bankData']['bankAccounts'][accIdx]['currentBalance']))
            balAvail = abs(float(jSonData['bankData']['bankAccounts'][accIdx]['availableBalance']))
            
            try:
                intRate = float(jSonData['bankData']['bankAccounts'][accIdx]['additionalDetails']['interestRate'])    
            except (ValueError, KeyError): 
                intRate = 21.00
            try:
                limit = float(jSonData['bankData']['bankAccounts'][accIdx]['additionalDetails']['creditLimit'])
            except (ValueError, KeyError): 
                limit = (balCurrOut) + (balAvail)
            try:
                minAmtDue = float(jSonData['bankData']['bankAccounts'][accIdx]['additionalDetails']['minimumAmountDue'])
            except (ValueError, KeyError): 
                minAmtDue = np.nan
            try:
                minAmtDueDate = jSonData['bankData']['bankAccounts'][accIdx]['additionalDetails']['minimumAmountDueDate']
            except (ValueError, KeyError): 
                minAmtDueDate = np.nan
      
            #calc obs commitment
            #limit position
            cntLim = round(limit * cntLimIntRate/100,2)
            #balance position
            cntBal = abs(round(np.pmt(intRate/100/12,cntBalMonths,balCurrOut,0),2))
            #sum
            cntObs = round(cntLim + cntBal,2)

            #calc obs average monthly repayment
            #grab txns
            txnObs = pd.DataFrame(jSonData['bankData']['bankAccounts'][accIdx]['transactions'])

            #if no repayment txns, then ratio is null
            cntRepayRatioObs = np.nan
            if len(txnObs) > 0:
                #get repayment txns: credits, recent, repayment narration detection
                #credits
                txnRepay = txnObs[txnObs['amount']>0].copy()
                txnRepay['text'] = txnRepay['text'].str.upper()
                #repayment narration detection
                txnRepay = txnRepay[txnRepay.text.str.contains("CREDIT CARD REPAYMENT|CREDIT CARD PAYMENT|CREDIT CARD PAYMEN|CC REPAYMENT|CC PAYMENT|CC AUTO PAYMENT|PAYMENT THANKYOU")]
                #some exist and recent
                if (len(txnRepay) > 0) and (np.datetime64(txnRepay.date.max()) >= (np.datetime64(appDate)-obsRecentRepayTxnBar)):
                    #wrangle commitment at point of repayment
                    txnRepay['monYY'] = pd.to_datetime(txnRepay['date'], format='%Y-%m-%d').dt.to_period("M")
                    txnRepay['balBeforeTxn'] = txnRepay.balance.astype(float) + txnRepay.amount.astype(float)
                    txnRepay['cnt'] = abs(np.pmt(intRate/100/12,cntBalMonths,txnRepay.balBeforeTxn,0)) + cntLim
                    txnRepay = txnRepay.groupby('monYY').agg({'amount':'sum','cnt':'sum'}) \
                                .merge( \
                                       txnRepay.groupby('monYY').agg({'amount':'sum','cnt':'sum'}).groupby('amount').agg({'amount':'count'}).rename(columns={'amount':'amountCount'}).reset_index() \
                                       ,how='left',on='amount')
                                
                    #exclude repayment anomalies i.e. principal payments - +- 2 standard deviations of repayment
                    #max procedure
                    if txnRepay[txnRepay.amount == txnRepay.amount.max()]['amountCount'].item() == 1:
                        if txnRepay.amount.max() > (np.std(txnRepay[txnRepay.amount != txnRepay.amount.max()].amount) * syph) + (txnRepay[txnRepay.amount != txnRepay.amount.max()].amount.mean()):
                            txnRepay = txnRepay[txnRepay.amount != txnRepay.amount.max()]
                    
                    #min procedure - may not need.
                    if txnRepay[txnRepay.amount == txnRepay.amount.min()]['amountCount'].item() == 1:
                        if txnRepay.amount.min() < (np.std(txnRepay[txnRepay.amount != txnRepay.amount.min()].amount) * syph) - (txnRepay[txnRepay.amount != txnRepay.amount.max()].amount.mean()):
                            txnRepay = txnRepay[txnRepay.amount != txnRepay.amount.min()]
                    
                    #calculate ratio: repayment amount/commitment 
                    cntRepayRatioObs = (txnRepay.amount/txnRepay.cnt).mean()
  
            #write observed CC + commitment deets to dataframe            
            tmpAppend = pd.DataFrame([[cntType,accType,accName,accNum,accBsb,balCurrOut,balAvail,limit,intRate,minAmtDue,minAmtDueDate,cntLim,cntBal,cntObs,cntRepayRatioObs]],columns=obsCcVar)           
            ccCntObs = ccCntObs.append(tmpAppend,ignore_index=True)

        #inferred CC from transaction and savings accounts
        #cycle through transaction/savings accounts
        if jSonData['bankData']['bankAccounts'][accIdx]['accountType'].upper().replace(" ", "") in 'TRANSACTION SAVINGS'.split():
            #grab transcations
            txnInf = pd.DataFrame(jSonData['bankData']['bankAccounts'][accIdx]['transactions'])
            
            #wrangle outbound CC repayment txns
            if len(txnInf) > 0:
                #debits
                txnInf = txnInf[txnInf['amount']<0]
                txnInf['text'] = txnInf['text'].str.upper()
                
                #get repayment types
                txnInfCcRepay = txnInf[txnInf.type.str.upper()=="CREDIT CARD REPAYMENTS"].copy()
                
                if len(txnInfCcRepay) > 0:
                    
                    #clean narrations
                    txnInfCcRepay['infNarration'] = txnInfCcRepay.text.apply(lambda x: re.sub("\d+", " ", x))
        
                    #aggregate by narration cashflow
                    infAgg = txnInfCcRepay.groupby('infNarration text'.split()).agg({'amount':'count sum'.split(),'date':'min max'.split()}).reset_index()
                    infAgg.columns = ["_".join(x) for x in infAgg.columns.ravel()]
         
                    #Recent txn threshold
                    infAgg = infAgg[infAgg.date_max.apply(lambda x: np.datetime64(x) >= (np.datetime64(appDate)-infRecentTxnBar))==True]          
                    
                    #at least 3 instances of repayment to be considered a commitment
                    infAgg = infAgg[(infAgg['amount_count']>infCountTxnBar)]
        
                    #exlcude failing txns
                    txnInfCcRepay = txnInfCcRepay[txnInfCcRepay.infNarration.isin(infAgg.infNarration_.tolist())]
        
                    #fuzzy deduplication of narrations;
                    infAggList = infAgg.infNarration_.tolist()
                    
                    #set up inferred fuzzy match table
                    infFuzz = pd.DataFrame([[0]*5], columns='idInf txt fuzzScore txtHit idInfHit'.split())
                             
                    tmpList = infAggList.copy()

                    #fuzzy de-duping
                    for idx, txt in enumerate(infAggList):
                        tmpList = [t for t in tmpList if t is not txt]
                                       
                        if len(tmpList) == 0:
                            break
                       
                        appendFuzz = pd.DataFrame(process.extract(txt,tmpList))
                        appendFuzz.rename(columns={0:'txtHit',1:'fuzzScore'},inplace=True)
                        appendFuzz['idxHit'] = appendFuzz.index
        
                        appendFuzz = appendFuzz[appendFuzz.fuzzScore >= fuzzThreshold]
                                       
                        if len(appendFuzz) == 0:
                            next
                        else: 
                            appendFuzz.loc[appendFuzz.fuzzScore >= fuzzThreshold,'idInf'] = idx
                            appendFuzz.loc[appendFuzz.fuzzScore >= fuzzThreshold,'txt'] = txt
                            infFuzz = infFuzz.append(appendFuzz,ignore_index=True)
                            tmpList = [t for t in tmpList if t not in appendFuzz.txtHit.tolist()]
                    
                    infFuzz = infFuzz[1:]

                    if len(infFuzz) > 0:
                        tmp0 = txnInfCcRepay.merge(infFuzz['idInf txt'.split()].drop_duplicates(['idInf','txt'],keep = 'first',inplace=False),how='left',left_on='infNarration',right_on='txt',suffixes=["",""])               
                        tmp1 = tmp0[tmp0.idInf.isnull()].drop('idInf',axis=1).merge(infFuzz['idInf txtHit'.split()].drop_duplicates(['idInf','txtHit'],keep = 'first',inplace=False),how='left',left_on='infNarration',right_on='txtHit',suffixes=["",""])
                        txnInfCcRepay = tmp0[tmp0.idInf.notnull()].append(tmp1,ignore_index=True).drop('txt',axis=1)
                    
                    else:
                        txnInfCcRepay['txtHit'] = np.nan
                        tmp0 = txnInfCcRepay.groupby('infNarration',as_index=False).agg({'amount':'mean'})
                        tmp0['idInf'] = tmp0.index
                        txnInfCcRepay = txnInfCcRepay.merge(tmp0['idInf infNarration'.split()],how='left',on='infNarration')            

                    #inferred CC deets 
                    infCc = txnInfCcRepay.drop_duplicates('idInf')['idInf text infNarration txtHit'.split()].copy()
    
                    #calculate average monthly repayment of inferred faciltiees
                    txnInfCcRepay['monYY'] = pd.to_datetime(txnInfCcRepay['date'], format='%Y-%m-%d').dt.to_period("M")
                                   
                    txnInfCcRepayMon = txnInfCcRepay.groupby('idInf monYY'.split()).agg({'amount':'sum'}).reset_index() \
                                        .merge(txnInfCcRepay.groupby('monYY').agg({'amount':'count'}).rename(columns={'amount':'amountCount'}).reset_index() \
                                               ,how='left',on='monYY')
        
                    #apply anomaly repayment exclusion
                    tmpSyph = pd.DataFrame([[0]*len(txnInfCcRepayMon.columns.values.tolist())],columns=txnInfCcRepayMon.columns.values.tolist())

                    for idx in txnInfCcRepayMon.idInf.unique().tolist():
                        tmp0 = txnInfCcRepayMon[txnInfCcRepayMon.idInf==idx]
        
                        if tmp0[tmp0.amount == tmp0.amount.max()]['amountCount'].item() == 1:
                            if tmp0.amount.max() > (np.std(tmp0[tmp0.amount != tmp0.amount.max()].amount) * syph) + (tmp0[tmp0.amount != tmp0.amount.max()].amount.mean()):
                                tmp0 = tmp0[tmp0.amount != tmp0.amount.max()]
                        
                        #may not need min procedure - or specific syph threshold.
                        if tmp0[tmp0.amount == tmp0.amount.min()]['amountCount'].item() == 1:
                            if tmp0.amount.min() < (np.std(tmp0[tmp0.amount != tmp0.amount.min()].amount) * syph) - (tmp0[tmp0.amount != tmp0.amount.max()].amount.mean()):
                                tmp0 = tmp0[tmp0.amount != tmp0.amount.min()]
        
                        tmpSyph = tmpSyph.append(tmp0,ignore_index=True)            
        
                    txnInfCcRepayMon = tmpSyph[1:].copy()
                    
                    infCc = infCc.merge(txnInfCcRepayMon.groupby('idInf',as_index=False).agg({'amount':'mean'}),how='left',on='idInf')            
                    infCc['amount'] = abs(infCc['amount'])
                    
                    #get tran/savings account deets
                    infCc['cntType'] = 'inf'
                    infCc['accName'] = jSonData['bankData']['bankAccounts'][accIdx]['accountName']
                    infCc['accType'] = jSonData['bankData']['bankAccounts'][accIdx]['accountType']
                    infCc['accNum'] = jSonData['bankData']['bankAccounts'][accIdx]['accountNumber']
                    infCc['accBsb'] = jSonData['bankData']['bankAccounts'][accIdx]['bsb']
                   
                    #write inferred CC + commitment deets to dataframe
                    ccCntInf = ccCntInf.append(infCc,ignore_index=True)

    #aggregate observed and inferred
    ccCntObs = ccCntObs[1:]
    ccCntInf = ccCntInf[1:]
    
    #average repayment across inferred CCs
#    cntRepayRatioObsMu = ccCntObs.cntRepayRatioObs.mean()   

    
    #apply repayRatio to inferred repayments to get commitment;
    #nan not resolving...
#    if cntRepayRatioObsMu != np.nan:
#        ccCntInf['cntInf'] = ccCntInf.apply(lambda x: x.amount/cntRepayRatioObsMu , axis=1)
        
        
    ccCntObs_jS = ccCntObs.to_json()
#    ccCntInf_jS = ccCntInf.to_json()
    
#    return(ccCntObs_jS,ccCntInf_jS)
    return(ccCntObs_jS)
  

if __name__ == '__main__':
    app.run()
  
#app.run(host='0.0.0.0', port= 8092)
