import sys
sys.path.append("../..")
from DeepNLP.model.DPOS import DPOS

if __name__ == '__name__':
    # POS tagging in Chinese
    model_path = '/data/Yangyang/D_project/saved_models/POS_baseline_cn_2021-07-09-14-58-20/model'
    deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=False, use_memory=True)
    sentence = [['中', '美', '在', '沪', '签订', '高', '科技', '合作', '协议'],
                ['新华社', '上海', '八月', '三十一日', '电', '（', '记者', '白国良', '、', '夏儒阁', '）']]
    result_list = deepnlp.predict(sentence_list=sentence)
    print(result_list)
    #[['中_NR', '美_NR', '在_P', '沪_NR', '签订_VV', '高_JJ', '科技_NN', '合作_NN', '协议_NN'],
    # ['新华社_NR', '上海_NR', '八月_NT', '三十一日_NT', '电_NN', '（_PU', '记者_NN', '白国良_NR', '、_PU', '夏儒阁_NR', '）_PU']]


    # POS tagging in English
    model_path = '/data/Yangyang/D_project/saved_models/POS_baseline_en_2021-07-15-20-28-44/model'
    deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=False, use_memory=True)
    sentence = [['The', 'bill', 'intends', 'to', 'restrict', 'the', 'RTC', 'to', 'Treasury', 'borrowings', 'only', ',', 'unless', 'the', 'agency', 'receives', 'specific', 'congressional', 'authorization', '.'],
                ['The', 'complex', 'financing', 'plan', 'in', 'the', 'S&L', 'bailout', 'law', 'includes', 'raising', '$', '30', 'billion', 'from', 'debt', 'issued', 'by', 'the', 'newly', 'created', 'RTC', '.']]
    result_list = deepnlp.predict(sentence_list=sentence)
    print(result_list)
    #[['The_DT', 'bill_NN', 'intends_VBZ', 'to_TO', 'restrict_VB', 'the_DT', 'RTC_NNP', 'to_TO', 'Treasury_NNP', 'borrowings_NNS', 'only_RB', ',_,', 'unless_IN', 'the_DT', 'agency_NN', 'receives_VBZ', 'specific_JJ', 'congressional_JJ', 'authorization_NN', '._.'],
    # ['The_DT', 'complex_JJ', 'financing_NN', 'plan_NN', 'in_IN', 'the_DT', 'S&L_NN', 'bailout_NN', 'law_NN', 'includes_VBZ', 'raising_VBG', '$_$', '30_CD', 'billion_CD', 'from_IN', 'debt_NN', 'issued_VBN', 'by_IN', 'the_DT', 'newly_RB', 'created_VBN', 'RTC_NNP', '._.']]


    # joint chinese word segmentation and POS tagging
    model_path='/data/Yangyang/D_project/saved_models/AC_SP_use_memory_2021-07-21-00-01-05/model'
    deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=True, use_memory=False)
    sentence = [
        ['法', '正', '研', '究', '从', '波', '黑', '撤', '军', '计', '划', '新', '华', '社', '巴', '黎', '９', '月', '１', '日', '电', '（',
         '记', '者', '张', '有', '浩', '）'],
        ['法', '国', '国', '防', '部', '长', '莱', '奥', '塔', '尔', '１', '日', '说', '，法', '国', '正', '在', '研', '究', '从', '波', '黑',
         '撤', '军', '的', '计', '划', '。']]
    result_list = deepnlp.predict(sentence_list=sentence)

    # [['法_NN', '正_NN', '研究_VV', '从_NN', '波黑_NR', '撤_VV', '军_NN', '计划_NN', '新华社_NN', '巴黎_NR', '９月_NT', '１_CD', '日_NT', '电_NN', '（_NN', '记者_NN', '张有浩_VV', '）_PU'],
    #  ['法国_NN', '国防_NN', '部_NN', '长_NN', '莱奥塔尔_NR', '１日_NT', '说_NN', '，法_PU', '国_NN', '正_AD', '在_AD', '研究_VV', '从_NN', '波黑_NR', '撤_VV', '军_NN', '的_NN', '计划_NN', '。_NN']