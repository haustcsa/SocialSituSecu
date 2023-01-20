# DBA_Casrel pytorch

参考文献
[1] Wei ZP, Su JL, Wang Y, et al. A Novel Cascade Binary Tagging Framework for Relational Triple Extraction[C]//Proc of the 58th Annual Meeting of the Association for Computational Linguistics, Stroudsburg, PA: Association for Computational Linguistics, 2020: 1476-1488
[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Proceeding of the 31st Conference on Neural Information Processing Systems, Long Beach, CA, USA: December 4-9, 2017. MIT Press: Cambridge, MA, 2017: 5998-6008.
主要功能
从自然语言文本中抽取出实体关系三元组,有效解决三元组重叠问题。
构建Duie_Bert预训练模型对输入的文本进行编码；
利用特定关系-实体向量引导的多头注意力机制来增强编码层输出向量的特征表达；
在此基础上，利用层叠式指针标注框架(CasRel)抽取出对应的尾实体，完成关系三元组的抽取。
