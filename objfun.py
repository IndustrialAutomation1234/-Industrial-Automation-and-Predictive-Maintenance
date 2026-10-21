import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_CSAViT_SeNet import Model_CSAViT_SeNet


def objfun(Soln):
    Data = Global_Vars.Images
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            learnperc = round(Data.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval, pred = Model_CSAViT_SeNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(pred, Test_Target)
            Fitn[i] = (1 / Eval[7])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        learnperc = round(Data.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Data[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Data[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval, pred = Model_CSAViT_SeNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(pred, Test_Target)
        Fitn = (1 / Eval[7])
        return Fitn
