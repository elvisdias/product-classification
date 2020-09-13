# Authors: Alysson Casimiro e Elvis Dias
# Licensed under the MIT License

"""
============================================================================================
Descrição Geral:
    Esse módulo implementa um pipeline de treinamento e predição na categorização de texto para fins de classificar produtos de 
    notas fiscais eletrônicas apenas por suas descrições de produtos (campo xprod) — visto que não há forma eficaz de classificar através de outros atributos preenchidos —
    e calcular o preço para cada classe de produto.
    Esse arquivo é escrito pensando no agendamento de tarefas automáticas, portanto é utilizado em conjunto de outros arquivos: Para sua execução mínima,
    é necessário variáveis globais com caminhos de diretório local e nomes de arquivos padrões a serem salvos e consumidos — foi usado um arquivo de configuração .ini
    Ainda, requer um arquivo csv contendo alguns valores mínimos contidos numa NFe padrão:
        - Descrição: 'xprod'
        - Valor Unitário: 'vuncom'
        - Data de Emissão: 'dhEmi'
        - NCM: 'ncm'
    Outros valores da nota fiscal são utilizados somente para padronizar arquivos de output. 
    As variáveis de configuração usadas — ignorando as de autenticação no banco — são:
        - path_csvs                           : caminho para salvar arquivos csvs com as descrições classificadas e preços calculados para cada classe de descrição encontrada
        - path_files_classificacao            : caminho para salvar arquivos de treinamento relativos ao modelo. Estes arquivos são consumidos pela fase de treinamento,
        - produtos_banco_filename             : nome do arquivo csv de input contendo dados da NFe
        - precos_filename                     : nome do arquivo csv de output para criar tabela de preços para classes para uma classe de produtos com dados padrão consumidos por um painel.
        - produtos_classificados_filename     : nome de arquivo pkl de output do treinamento contendo todos os dados com seu histórico de classificação do pipeline. Esse arquivo é usado na fase de treinamento.
        - data_ultima_classificacao           : nome do arquivo pkl de output com a data em formato timestamp da data mais recente contendo produtos classificados, usado para filtrar produtos na próxima execução.
        - clf_filename                        : nome do arquivo pkl de output do modelo treinado, usado para predição de classes.
        - classes_treinamento                 : nome do arquivo pkl de output fornecido pela etapa de treinamento, contendo as classes que o modelo reconhece, usado na etapa de predição.
        - count_vect_filename                 : nome do arquivo pkl do arquivo de vetorização treinado, utilizado para vetorizar novos dados na predição.
        - tfidf_transformer_filename          : nome do arquivo pkl do arquivo de tfidf treinado, para alterar vetorização de novos dados na predição.
        - label_encoder_filename              : nome do arquivo pkl do arquivo de codificação treinado, para alterar texto da etapa de pré-classificação de novos dados na predição.
===========================================================================================
Usabilidade:
	O algoritmo pode ser pensado em três etapas: limpeza das descrições originais, classificação "manual" baseada no ncm e aplicação no modelo. 
    O construtor da classe é responsável por inicializar variáveis que filtram as notas fiscais e ajudam na sua limpeza. Dentre estas, para futuros usos é necessária manutenção apenas
    nas lista de NCM (expandir pré-classificação) e lista de stopwords. 
	Etapas:
    1. A etapa de limpeza pode vir a mudar de acordo com novos produtos contendo regalias próprias, porém as limpezas feitas pelos métodos "limpa_ucom" e "limpa_descricao"
       fazem limpezas universais.
    2. A classificação manual necessária para todo produto que for ser usado (com ou sem modelo), segue um padrão e exige pouca novidade de um produto para outro, no método "classifica_descricoes".
	3. O treinamento e predição de modelo usam métodos semelhantes, tendo que fazer pré-processamento de PLN e de vetorização de texto antes de aplicar nos modelos. Essa étapa segue pipelines específicos: "exec_fluxo_predicao" e "exec_fluxo_treinamento"    
	- Treinamento
		:lê csv de produtos aplicas limpezas em unidades e descrições, cria coluna "descrição" com classes em texto, aplica métodos de pln nas descrições, encoda a coluna "descrição", aplica modelos de vetorização de texto e aplica ao modelo o texto e a "descrição" encodados executando o treinamento.
	- Predição
		:lê csv de produtos aplicas limpezas em unidades e descrições, cria coluna "descrição" com classes em texto, aplica métodos de pln nas descrições, encoda a coluna "descrição", aplica modelos de vetorização de texto e aplica ao modelo o texto e a "descrição" encodados para saber qual porcetagem
         de cada descrição ser de uma classe conhecida pelo modelo. Caso o modelo achei que tem mais de 90% (valor de threshold) de chance, a classe é aplicada à descrição
        :Nessa etapa pode ser levado em conta o modelo ou não. Sem o modelo há mais chance de atingir 0% de falsos positivos, uso mais crítico com produto classificados apelas pela fase de pré-classificação no método "classifica_descricoes", enquanto que com o modelo há a % de erro que influenciará no preço final.
    Variáveis:
        class_size(int)   : número de samples mínimos que uma classe precisa ter pós pré classificação para ser considerada pelo modelo
        save(bool)        : variável boolean que determina se os métodos irão gerar arquivos de output. (padrão True)
        threshold(number) : valor mínimo de semelhança de uma descrição com uma classe dada pelo modelo para que a descrição se torne daquela classe
        df_aux(pandas df) : dataframe contendo descrições não utilizadas pela etapa de treinamento, usada para testar o modelo em descrições não vistas.
        sample(str)       : método de sampling para aplicar sobre a quantidade de descrições que irão para o modelo de treinamento. ("over" ou "under")
        method(str)       : fluxo seguido pelo módulo, treino ou predição, "Train" ou "Predict" (default). Altera filtragem de produtos.
"""
# !pip install imbalanced-learn
import os
import re
import nltk
import pickle
import pandas as pd
import numpy as np
import string
import unidecode
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from string import punctuation
from sqlalchemy import create_engine
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict,\
                         ShuffleSplit,StratifiedShuffleSplit, KFold, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('rslp')

#from ciex.painelpreco.config_nfe import config_nfe
from config_nfe import *

class ClassificarProdutos():

    def __init__(self):
        self.config_db = config_nfe['POSTGRESQL']
        self.config_files = config_nfe['FILES']
        self.path_files = self.config_files['path_files_classificacao']
        self.path_csvs = self.config_files['path_csvs']

        engine = create_engine("postgresql+psycopg2://{}:{}@{}:{}/{}".format(
                                            self.config_db['user'],
                                            self.config_db['pass'],
                                            self.config_db['host'],
                                            self.config_db['port'],
                                            self.config_db['database']),
                                            connect_args={'options': '-csearch_path={}'.format(self.config_db['schema'])}) 
        self.conn = engine.connect()
        self.df_produto = pd.read_sql("select nfe.idnfe, nfe.dhEmi, dest.xnome, dest.enderdest_cmun, p.NCM, p.cEAN, p.ucom, p.qcom, p.xProd, p.vuncom, p.vprod\
                                        from produto as p\
                                        inner join nfe on p.idNFe = nfe.idNFe\
                                        inner join destinatario as dest on dest.idNFe = nfe.idNFe\
                                        inner join resumo_nfe as resumo on resumo.chave_nfe = nfe.idNFe\
                                        where resumo.situacao_nfe = 'NFE AUTORIZADA'", self.conn)
        
        self.precos_filename = self.config_files['precos_filename']
        self.produtos_classificados_filename = self.config_files['produtos_classificados_filename']
        self.unis = ['x','mm','m','cm','ml','g','mg','kg','unidade','unidades','polegada','polegadas','grama','gramas','gb','mb',\
                'l','li','lts','lr','lt','ltrs', 'ltr','litro','litros','mts','un','mgml','w','hz','v','gr','lonas','cores','mcg']
        self.lista_lt = ['li','lit','litro', 'litros', 'ltr', ' l ','lt', 'ltrs']
        self.lista_kg = ['kg','quilo','km','kg1', 'kg3', 'kg2']
        self.lista_und = ['und','un','un0001','unid','un1','1', 'uni','un.','unid.','unidad','ud','lata','lat', 'embal','garraf','garrafa','rl']
        self.lista_pct = ['pct','pc','pt','pc1','pact','pacote','pa','pcs']
        self.lista_cx = ['cx','caixa','cx20','cx12','cx48','cxs','cxa','cx24']
        self.lista_frd = ['fdo','fardo','frd','fd','fd12','fr','far','fd27','fdo']
        self.lista_geral = ['lt','kg','und','pct','cx','frd', 'sc']
        self.lista_saco = ['sc']
        self.unidades = set(self.unis+self.lista_lt+self.lista_kg+self.lista_und+self.lista_pct+self.lista_cx+self.lista_frd+self.lista_saco\
            +['g','gr', 'ml','cm'])
        
        self.lista_ncms = set(['02012090', '02013000', '02022090', '02071200', '02071300', '02071400', '02102000', '04022110', \
            '04031000', '04039000', '07061000', '07061000', '09012100', '10062010', '10063011', '10063021', '11010010', '11022000', \
            '11041200', '11041900', '11042300', '11062000', '11081200', '11081400', '15079011', '15091000', '15171000', '17019900', \
            '17031000', '17039000', '18061000', '18069000', '19021900', '19053100', '19059020', '20060000', '20083000', '20089900', \
            '20099000', '21042000', '21069090', '22011000', '25010020', '25221000', '25222000', '25223000', '25232910', '27101259', \
            '27101921', '27111910', '28044000', '28111990', '28112990', '28289011', '28539090', '30012090', '30021229', '30021590', \
            '30032029', '30032099', '30039071', '30041019', '30042029', '30042099', '30049024', '30049037', '30049059', '30049069', \
            '30049099', '30059090', '30051090', '34022000', '36050000', '39241000', '40111000', '40112090', '40151100', '40151900', \
            '48181000', '48183000', '48189090', '84151011', '84212300', '96039000'])
        self.pt_stopwords = set(nltk.corpus.stopwords.words("portuguese"))
        punct = set(list(punctuation))
        
        stopwords = set(['gtin','ok','marca','in', 'natura', 'nacional','embalagem','vitamassa','tipo','hort','qtd.', '1ª',\
                 'qualidade', 'bom', 'todo','cs','dona', 'clara','rei','ouro','lua', 'azul','belo','grao','marata',\
                 'polo', 'ii', 'iii','bem','primeira','endereco'])
        self.pt_stopwords.update(punct)
        self.pt_stopwords.update(stopwords)

    @staticmethod
    def grava(obj, path, filename):
        '''
		Método que salva arquivo no formato .pkl 
		Args:
			obj: objeto, qualquer formato, a ser salvo
            path: caminho para salvar objeto
            filename: nome do arquivo pkl
		
		'''
        pkl_file = open(path + filename, 'wb')
        pickle.dump(obj, pkl_file)
        pkl_file.close()
	
    @staticmethod
    def abre(path, filename):
        '''
        Método que abre arquivo no formato .pkl no formato original
		Args:
            path: caminho onde está objeto .pkl
            filename: nome do arquivo pkl
		
		'''

        pkl_file = open(path + filename, 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()
        return obj

    @staticmethod
    def resampling(x_train, y_train, sample='over'):
        '''
        Realiza over/undersampling da quantidade de descrições usadas no treinamento através da biblioteca imblearn. 
		Args:
			x_train(sparse matrix): matrix de descrições vetorizadas.
            y_train(pandas.dataframe): classes de cada descrição vetorizada.
            method(str): método a ser usado, over ou under.
		Retorna:
			x_train(sparse matrix): matrix de descrições vetorizadas resampled
            y_train(pandas.dataframe): classes de cada descrição vetorizada resampled
        '''
        if sample != 'over':
            rus = RandomUnderSampler(replacement=False)
            x_train_under, y_train_under = rus.fit_sample(x_train, y_train)

            return x_train_under, y_train_under
        
        rus = RandomOverSampler()
        x_train_over, y_train_over = rus.fit_sample(x_train, y_train)  

        return x_train_over, y_train_over
   
    @staticmethod
    def visualiza_distribuicao_original_classes(df):
        '''
        Plota em barras quantidade de valores por classe pré-classificada. Ideal para usar para análise de quantidades do df alterando o groupby
        para analisar diferentes proporções do df.
		Args:
			df(pandas.DataFrame): dataframe de produtos e classes.
        '''
        fig = plt.figure(figsize=(20,5))
        df.groupby('definicao').definicao.count().plot.bar(ylim=0)
        fig.set_facecolor('xkcd:white')
        plt.show()

    @staticmethod
    def visualiza_resultado(y_test, y_pred):
        '''
        Plota matriz de confusão para comparar predição do modelo com valor real de teste.
		Args:
			y_test(list): classes de teste para descrições do df de teste.
            y_pred(list): classes previstas pelo modelo para o df de teste
        '''
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12,8))
        sns.heatmap(cm, annot=True, cbar=False, fmt='g')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Accuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
        plt.show()

    @staticmethod
    def metodos_pln(df):
        '''
        Aplica métodos de PLN de steeming e lematização sobre as descrições de produtos.
		Args:
			df(pandas.DataFrame): Contendo coluna 'xprod'.
		Retorna:
			df(pandas.DataFrame): Contendo coluna 'xprod' modificada.
        '''
        stemmer = nltk.stem.RSLPStemmer()
        lemmatizer=WordNetLemmatizer()
       
        df['xprod'] = df['xprod'].apply(lambda x: ' '.join([lemmatizer.lemmatize(item) for item in x.split() ] ))
        df['xprod'] = df['xprod'].apply(lambda x: ' '.join([stemmer.stem(item) for item in x.split()] ))
        
        return df
    
    @staticmethod
    def calc_media_saneada(df):
        '''
        Realiza calculo de média saneada, método recomendado para cálculo de preços médios, para um dataframe dado.
		Args:
			df(pandas.DataFrame): Contendo pelo menos uma coluna 'Valor Unitario de Comercializacao' do tipo numerico
		Retorna:
			df(pandas.DataFrame): Valores de média saneada, máxima e minimo para preços de uma coluna.
        '''
        preco = df['vuncom']
        preco_max = max(preco)
        preco_min = min(preco)
        preco_mean = preco.mean()
        LS = preco_mean + preco.std()
        LI = preco_mean - preco.std()		
        CV = abs(preco.std() / preco_mean)

        if CV > 0.25:
            precos = preco[preco.between(LI, LS)]
            return (precos.mean(), preco_max, preco_min)
        else:
            return (preco_mean, preco_max, preco_min)

    @staticmethod
    def reindex_grupos(df):
        '''
        Faz um mapeamento dos numeros de grupos determinados pelo algoritmo com um numero sequencial, para fins de melhor usabilidade.
		Args:
			df(pandas.dataframe): dataframe de produtos contendo coluna 'classe'
		Retorna:
			df(pandas.dataframe): dataframe de produtos contendo novos valores na coluna 'classe'
        '''
        classes = df['classe'].unique().tolist()
        classes.sort()
        reindex={}
        
        for i, classe in enumerate(classes):
            reindex[classe] = i+1
        
        df['classe'] = df['classe'].map(reindex)
        
        return df

    @staticmethod
    def classifica_descricoes(df):
        '''
        Realiza pré-classificação de descrições a partir de padrões regex e NCMs
		Args:
			df(pandas.DataFrame): Contendo pelo menos uma coluna 'xprod' e 'ncm' do tipo string
		Retorna:
			df(pandas.DataFrame): Retorna o mesmo objeto recebido, com nova coluna 'definicao' contendo classe em texto para cada produto ou NaN para produtos que não se encaixaram em classes.
        '''
        df['definicao'] = np.NaN
        
        """  ################## COMBUSTIVEIS   ##################   """

        # Diesel s-10 Aditivado
        df.loc[(df.ncm.isin(['27101921','27101259'])) &  
                    (
                        ( (df.xprod.str.contains('diesel')) & 
                            (df.xprod.str.contains('adit'))  ) |
                        ( (df.xprod.str.contains('diesel')) &
                            (df.xprod.str.contains('10'))  )
                    ) & (df.definicao.isnull()), 'definicao'] = 'diesel s-10'
        # Diesel s-500 Aditivado
        df.loc[(df.ncm.isin(['27101921', '27101259'])) &  
                    (df.xprod.str.contains('500')) & 
                    (df.xprod.str.contains('adit')), 'definicao'] = 'diesel s-500'
        # Diesel s-500 Comum
        df.loc[(df.ncm.isin(['27101921', '27101259'])) &  
                    (df.xprod.str.contains('500')) & 
                    (df.definicao.isnull()), 'definicao'] = 'diesel s-500'
        # Tratando outros casos de diesel comum
        df.loc[(df.ncm.isin(['27101921', '27101259'])) &  
                    (df.xprod.str.contains('comum')) & 
                    (df.xprod.str.contains('diesel')) &
                    (df.definicao.isnull()), 'definicao'] = 'diesel s-500'
        # Diesel s-500 Comum
        df.loc[(df.ncm.isin(['27101921', '27101259'])) &  
                    (df.xprod.str.contains('diesel')) & 
                    (df.definicao.isnull()), 'definicao'] = 'diesel s-500'
        # Gasolina Grid
        df.loc[(df.ncm.isin(['27101259', '27101921'])) &  
                    (df.xprod.str.contains('gasolina')) & 
                    (df.xprod.str.contains('grid')) &
                    (df.definicao.isnull()), 'definicao'] = 'gasolina aditivada'
        # Gasolina Aditivada
        df.loc[(df.ncm.isin(['27101259', '27101921'])) &  
                    (df.xprod.str.contains('gasolina')) & 
                    (df.xprod.str.contains('adit')) &
                    (df.definicao.isnull()), 'definicao'] = 'gasolina aditivada'
        # Gasolina Comum 
        df.loc[(df.ncm.isin(['27101259', '27101921'])) &  
                    ((df.xprod.str.contains('gasolina')) & 
                    (df.xprod.str.contains('comum'))) |
                    (df.xprod.str.contains('gasolina')) & 
                    (df.definicao.isnull()), 'definicao'] = 'gasolina comum'
        # Etanol
        df.loc[(df.ncm.isin(['27101259', '27101921'])) &  
                    (df.xprod.str.contains('etanol')) & 
                    (df.definicao.isnull()), 'definicao']= 'etanol comum'
        # botijao de gas de cozinha 45kg
        df.loc[(df.ncm.isin(['27111910'])) & 
                    (df.xprod.str.contains('45')) &
                    (df.xprod.str.contains('glp|gas', regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'botijao de gas de cozinha 45kg'
        # botijao de gas de cozinha 13kg
        df.loc[(df.ncm.isin(['27111910'])) & 
                    ((df.xprod.str.contains('13')) &
                    (df.xprod.str.contains('glp|gas', regex=True))) |
                    (df.xprod.str.contains('^glp$', regex=True)) |
                    ((df.xprod.str.contains('glp|gas', regex=True)) &
                    (df.xprod.str.contains('cozinha'))) &
                    (df.definicao.isnull()), 'definicao'] = 'botijao de gas de cozinha 13kg'


        """  ################## CARNES E FRANGOS   ##################   """

        # peito de frango 1kg
        df.loc[(df.ncm.isin(['02071400','02071300','02102000','02013000'])) &   
               (df.xprod.str.contains('pto|peito', regex=True)) &
               (df.xprod.str.contains('frango|fgo',regex=True))  & 
               (df.definicao.isnull()), 'definicao'] = 'peito de frango'
        # coxa e sobrecoxa de frango 1kg
        df.loc[(df.ncm.isin(['02071400', '02071200', '02071300','02102000'])) &  
                    (df.xprod.str.contains('coxa')) &
                    (~df.xprod.str.contains('carne')) &
                    (df.definicao.isnull()), 'definicao'] = 'coxa e sobrecoxa de frango'
        # coxa e sobrecoxa de frango 1kg
        df.loc[(df.ncm.isin(['02071400', '02071200', '02071300','02102000'])) &  
                    (df.xprod.str.contains('frango|fgo',regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'frango inteiro'
        # carne de charque
        df.loc[(df.ncm.isin(['02102000','02013000'])) &  
                    (df.xprod.str.contains('carne')) &
                    (df.xprod.str.contains('charq')) &
                    (df.definicao.isnull()), 'definicao'] = 'carne de charque'
        # carne de sol
        df.loc[(df.ncm.isin(['02102000','02013000'])) &  
                    (df.xprod.str.contains('carne')) &
                    (df.xprod.str.contains('sol')) &
                    (df.definicao.isnull()), 'definicao'] = 'carne de sol'
        # carne moida
        df.loc[(df.ncm.isin(['02012090', '02013000','02022090', '02102000'])) &  
                    (df.xprod.str.contains('carne|bovin',regex=True)) &
                    (df.xprod.str.contains('moid')) &
                    (df.definicao.isnull()), 'definicao'] = 'carne moida'               
        # costela bovina
        df.loc[(df.ncm.isin(['02012090', '02013000','02022090', '02102000'])) &  
                    (df.xprod.str.contains('costela')) &
                    (df.xprod.str.contains('bov|carne', regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'costela bovina'
        # carne bovina
        df.loc[(df.ncm.isin(['02012090', '02013000','02022090', '02102000'])) &  
                    (df.xprod.str.contains('carne')) &
                    (df.xprod.str.contains('bov', regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'carne bovina'


        """  ################## BISCOITOS E MACARRAO   ##################   """


        # biscoito cream cracker
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
                    (
                        (
                            ((df.xprod.str.contains('cream|crem', regex=True)) &
                            (df.xprod.str.contains('crack|crak', regex=True))) | 
                            (df.xprod.str.contains('crack|crak', regex=True))
                        ) |
                        (
                            (df.xprod.str.contains('agua')) &
                            (df.xprod.str.contains('sal'))
                        )
                    ) &
                (df.definicao.isnull()), 'definicao'] = 'biscoito cream cracker'
        # biscoito maizena 400g
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
                    (df.xprod.str.contains('maizena|maisena',regex=True)) &
                    (df.xprod.str.contains('400')) &
                    (df.definicao.isnull()), 'definicao'] = 'biscoito maizena 400g'
        # biscoito maizena 360g
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
                    (df.xprod.str.contains('maizena|maisena',regex=True)) &
                    (df.xprod.str.contains('360')) &
                    (df.definicao.isnull()), 'definicao'] = 'biscoito maizena 360g'
        #biscoito doce rosquinha 400g
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
               (df.xprod.str.contains('rosquinha')) &
               ((df.xprod.str.contains('400'))  |(~df.xprod.str.contains('\d+'))) &
               (df.definicao.isnull()), 'definicao'] = 'biscoito rosquinha 400g'
        # biscoito maria 400g
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
                    (df.xprod.str.contains('maria')) &
                    ((df.xprod.str.contains('400'))  |(~df.xprod.str.contains('\d+'))) &
                    (df.definicao.isnull()), 'definicao'] = 'biscoito maria 400g'
        # biscoito maria estrela 500g
        df.loc[(df.ncm.isin(['19053100','19059020'])) &  
               (df.xprod.str.contains('maria')) &
               (df.xprod.str.contains('500')) &
               (df.definicao.isnull()), 'definicao'] = 'biscoito maria 500g'
        # macarrao espaguete 500g
        df.loc[(df.ncm.isin(['19021900'])) & 
                            (df.xprod.str.contains('mac')) &
                            (df.xprod.str.contains('espag|spaghet', regex=True)) &
                            ((~df.xprod.str.contains('kg|\d+')) | 
                            (df.xprod.str.contains('500'))) &
                            (df.definicao.isnull()), 'definicao'] = 'macarrao espaguete 500g'
        # macarrao integral 500g
        df.loc[(df.ncm.isin(['19021900'])) & 
                    (df.xprod.str.contains('mac')) &
                    (df.xprod.str.contains('integral', regex=True)) &
                    ((~df.xprod.str.contains('kg|\d+')) | 
                    (df.xprod.str.contains('500'))) &
                    (df.definicao.isnull()), 'definicao'] = 'macarrao integral 500g'
        # macarrao integral 200g
        df.loc[(df.ncm.isin(['19021900'])) & 
                    (df.xprod.str.contains('mac')) &
                    (df.xprod.str.contains('integral', regex=True)) &
                    (df.xprod.str.contains('200')) &
                    (df.definicao.isnull()), 'definicao'] = 'macarrao integral 200g'

        """  ################## AGUA MIN E REFRI   ##################   """

        # agua 500ml
        df.loc[(df.ncm.isin(['22011000'])) &  
                    (df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('500 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'agua mineral 500ml'
        # agua 510ml
        df.loc[(df.ncm.isin(['22011000'])) &  
                    (df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('510 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'agua mineral 510ml'
        # agua 300ml
        df.loc[(df.ncm.isin(['22011000'])) &
               (df.xprod.str.contains('agua')) &
               (df.xprod.str.contains('300 ml')) &
               (df.definicao.isnull()), 'definicao'] = 'agua mineral 300ml'
        # agua 350ml
        df.loc[(df.ncm.isin(['22011000'])) &
               (df.xprod.str.contains('agua')) &
               (df.xprod.str.contains('350 ml')) &
               (df.definicao.isnull()), 'definicao'] = 'agua mineral 350ml'
        # agua 200ml
        df.loc[(df.ncm.isin(['22011000'])) &
                    (df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('200 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'agua mineral 200ml'
        # agua 1,5lt
        df.loc[(df.ncm.isin(['22011000'])) &    
               (df.xprod.str.contains('agua')) &
               (df.xprod.str.contains('1,5|1.5',regex=True)) &
               (df.definicao.isnull()), 'definicao'] = 'agua mineral 1,5lt'
        # agua 20L
        df.loc[(df.ncm.isin(['22011000'])) &  
                    ((df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('20 lt'))) |
                    ((df.xprod.str.contains('garraf')) &
                    (df.xprod.str.contains('20'))) &
                    (~df.xprod.str.contains('^envase', regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'agua mineral 20lt'
        # refrigerante 2lt
        df.loc[(df.ncm.isin(['22011000'])) &
                    (df.xprod.str.contains('refr')) &
                    (df.xprod.str.contains('2 lt')) &
                    (df.definicao.isnull()), 'definicao'] = 'refrigerante 2lt'

        
        """  ################## DESCARTAVEIS   ##################   """

        # copo descartavel 50ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &  
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains(' 50 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 50ml'
        # copo descartavel 150ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains('150 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 150ml'
        # copo descartavel 180ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &  
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains('180 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 180ml'
        # copo descartavel 200ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &  
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains('200 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 200ml'
        # copo descartavel 250ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &  
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains('250 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 250ml'
        # copo descartavel 300ml
        df.loc[(df.ncm.isin(['39241000', '22011000'])) &  
                    (df.xprod.str.contains('^copo', regex=True)) &
                    (df.xprod.str.contains('300 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'copo descartavel 300ml'


        """  ################## LIMPEZA   ##################   """

        # agua sanitaria 5lt
        df.loc[(df.ncm.isin(['28289011','96039000'])) &  
               (
                   ((df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('sanit'))) |
                    (df.xprod.str.contains('hipoclorito de sodio'))
               ) &
               (df.xprod.str.contains('5 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'agua sanitaria 5lt' 
        # agua sanitaria 2lt
        df.loc[(df.ncm.isin(['28289011','96039000'])) &  
               (
                   ((df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('sanit'))) |
                    (df.xprod.str.contains('hipoclorito de sodio'))
               ) &
               (df.xprod.str.contains('2 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'agua sanitaria 2lt' 
        # agua sanitaria 1lt
        df.loc[(df.ncm.isin(['28289011','96039000'])) &  
               (
                   ((df.xprod.str.contains('agua')) &
                    (df.xprod.str.contains('sanit'))) |
                    (df.xprod.str.contains('hipoclorito de sodio'))
               ) &
               (df.definicao.isnull()), 'definicao'] = 'agua sanitaria 1lt'
        # detergente 500g
        df.loc[(df.ncm.isin(['34022000'])) &  
                (df.xprod.str.contains('deterg|det |sabao liq', regex=True)) &
               (df.xprod.str.contains('500 g')) &
               (df.definicao.isnull()), 'definicao'] = 'detergente 500g'
        # detergente 500ml
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('deterg|det |lava louca|sabao liq', regex=True)) &
               ((df.xprod.str.contains('500')) | (~df.xprod.str.contains('\d+'))) &
               (df.definicao.isnull()), 'definicao'] = 'detergente 500ml'
        # detergente 5lt
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('deterg|det |lava louca|sabao liq', regex=True)) &
               (df.xprod.str.contains('5 lt|5000 ml')) &
               (df.definicao.isnull()), 'definicao'] = 'detergente 5lt'
        # detergente 2lt
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('deterg|det |lava louca|sabao liq', regex=True)) &
               (df.xprod.str.contains('2 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'detergente 2lt'
        # detergente 1lt
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('deterg|det |lava louca|sabao liq', regex=True)) &
               (df.xprod.str.contains('1 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'detergente 1lt'
        # sabao em po 500g
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('sabao')) &
               (df.xprod.str.contains('500 g')) &
               (df.definicao.isnull()), 'definicao'] = 'sabao em po 500g'
        # sabao em po 400g
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('sabao')) &
               (df.xprod.str.contains('400 g')) &
               (df.definicao.isnull()), 'definicao'] = 'sabao em po 400g'
        # sabao em po 1kg 
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('sabao')) &
               (df.xprod.str.contains('1 kg')) &
               (df.definicao.isnull()), 'definicao'] = 'sabao em po 1kg'
        # sabao em po 2kg 
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('sabao')) &
               (df.xprod.str.contains('2 kg')) &
               (df.definicao.isnull()), 'definicao'] = 'sabao em po 2kg'
        # sabao em po 5kg 
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('sabao')) &
               (df.xprod.str.contains('5 kg')) &
               (df.definicao.isnull()), 'definicao'] = 'sabao em po 5kg'
        # limpador 500ml
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('limpa|limp |veja',regex=True)) &
               (df.xprod.str.contains('500')) &
               (df.definicao.isnull()), 'definicao'] = 'limpador 500ml'
        # desinfetante 500ml
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('desinf'))&
               (df.xprod.str.contains('500')) &
               (df.definicao.isnull()), 'definicao'] = 'desinfetante 500ml'
        # desinfetante 1lt
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('desinf'))&
               (df.xprod.str.contains('1 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'desinfetante 1lt'
        # desinfetante 2lt
        df.loc[(df.ncm.isin(['34022000'])) &  
               (df.xprod.str.contains('desinf'))&
               (df.xprod.str.contains('2 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'desinfetante 2lt'


        """  ################## ALCOOL   ##################   """


        # alcool 70 5lt
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
               (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('70')) &
              (~df.xprod.str.contains('gel')) &
              (df.xprod.str.contains('5 lt')) &
              (df.definicao.isnull()), 'definicao'] = 'alcool 70% 5lt'
        # alcool 70 5lt
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
               (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('70')) &
              (df.xprod.str.contains('gel')) &
              (df.xprod.str.contains('5 lt')) &
              (df.definicao.isnull()), 'definicao'] = 'alcool gel 70% 5lt'
        # alcool gel 500ml
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
              (df.xprod.str.contains('alcool')) &
              ((df.xprod.str.contains('500')) |(~df.xprod.str.contains('lt'))) &
              (df.xprod.str.contains('gel')) &
              ((df.xprod.str.contains('70')) |
              (~df.xprod.str.contains('92|96|46|54|64|99', regex=True))) &
              (df.definicao.isnull()), 'definicao'] = 'alcool gel 70% 500ml'
        # alcool 70 500ml
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
              (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('500')) &
              ((df.xprod.str.contains('70')) |
              (~df.xprod.str.contains('92|96|46|54|64|99', regex=True))) &
              (~df.xprod.str.contains('vidro|veja',regex=True)) &
              (df.definicao.isnull()), 'definicao'] = 'alcool 70% 500ml'
        # alcool gel 64% 500ml
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
              (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('500')) &
              (df.xprod.str.contains('64')) &
              (df.xprod.str.contains('gel')) &
              (df.definicao.isnull()), 'definicao'] = 'alcool gel 64% 500ml'
        # alcool gel 46% 500ml
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
              (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('500')) &
              (df.xprod.str.contains('46')) &
              (df.xprod.str.contains('gel')) &
              (df.definicao.isnull()), 'definicao'] = 'alcool gel 46% 500ml'
        # alcool gel 70 1lt 
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
              (df.xprod.str.contains('alcool')) &
              (df.xprod.str.contains('gel')) &
              ((df.xprod.str.contains('70'))|
              (~df.xprod.str.contains('92|96|46|54|99',regex=True))) &
              (~df.xprod.str.contains('1 lt')) &
              (df.definicao.isnull()), 'definicao'] = 'alcool gel 70% 1lt'
        # alcool 70 1lt
        df.loc[(df.ncm.isin(['96039000', '28289011', '30049099'])) &  
               (df.xprod.str.contains('alcool')) &
              ((df.xprod.str.contains('70'))|
              (~df.xprod.str.contains('92|96|46|54|99',regex=True))) &
              ((df.xprod.str.contains('1 lt')) | (~df.xprod.str.contains('lt|ml',regex=True))) &
              (df.definicao.isnull()), 'definicao'] = 'alcool 70% 1lt'


        """  ################## OXIGENIO   ##################   """


        # Oxigênio medicinal 10m3
        df.loc[(df.ncm.isin(['28044000'])) &
                    (df.xprod.str.contains('oxi')) &
                    (df.xprod.str.contains('10 m')) &
                    (df.xprod.str.contains('med')) &
                    (df.definicao.isnull()), 'definicao'] = 'oxigênio medicinal 10m3'
        # Oxigênio medicinal 7m3
        df.loc[(df.ncm.isin(['28044000'])) &
                            (df.xprod.str.contains('oxi')) &
                            (df.xprod.str.contains('7 m')) &
                            (df.xprod.str.contains('med')) &
                            (df.definicao.isnull()), 'definicao'] = 'oxigênio medicinal 7m3'
        # oxido nitroso medicinal 33kg
        df.loc[(df.ncm.isin(['28111990', '28112990'])) &
                            (df.xprod.str.contains('oxido')) &
                            (df.xprod.str.contains('nit')) &
                            (df.xprod.str.contains('33 kg')) &
                            (df.definicao.isnull()), 'definicao'] = 'oxido nitroso medicinal 33kg'
        # oxido nitroso medicinal 28kg
        df.loc[(df.ncm.isin(['28111990', '28112990'])) &
                            (df.xprod.str.contains('oxido')) &
                            (df.xprod.str.contains('nit')) &
                            (df.xprod.str.contains('28 kg')) &
                            (df.definicao.isnull()), 'definicao'] = 'oxido nitroso medicinal 28kg'
        # oxido nitroso medicinal 14kg
        df.loc[(df.ncm.isin(['28111990', '28112990'])) &
                            (df.xprod.str.contains('oxido')) &
                            (df.xprod.str.contains('nit')) &
                            (df.xprod.str.contains('14 kg')) &
                            (df.definicao.isnull()), 'definicao'] = 'oxido nitroso medicinal 14kg'
        # ar comprimido 7m3
        df.loc[(df.ncm.isin(['28539090'])) &
                            (df.xprod.str.contains('ar')) &
                            (df.xprod.str.contains('comprimido')) &
                            (df.xprod.str.contains('7 m')) &
                            (df.definicao.isnull()), 'definicao'] = 'ar comprimido 7m3'
        # ar comprimido 10m3
        df.loc[(df.ncm.isin(['28539090'])) &
                            (df.xprod.str.contains('ar')) &
                            (df.xprod.str.contains('comprimido')) &
                            (df.xprod.str.contains('10 m')) &
                            (df.definicao.isnull()), 'definicao'] = 'ar comprimido 10m3'


        """  ##################  SAL   ##################   """


        # sal refinado 1kg
        df.loc[(df.ncm.isin(['25010020','34022000'])) &
                    (  (df.xprod.str.contains('sal ')) &
                        (df.xprod.str.contains('ref|1 kg|iod',regex=True))
                    ) | (df.xprod.str.contains('^sal$|^sal ', regex=True)) &
               (df.definicao.isnull()), 'definicao'] = 'sal refinado 1kg'
        

        """  ################## CONSTRUÇÃO   ##################   """


        # cimento 50kg
        df.loc[(df.ncm.isin(['25232910'])) &
               (df.xprod.str.contains('cimento|cp ii',regex=True)) &
               ((df.xprod.str.contains('50')) |
               (~df.xprod.str.contains('kg|\d+', regex=True))) &
               (df.definicao.isnull()), 'definicao'] = 'cimento 50kg'
        # cimento 40kg
        df.loc[(df.ncm.isin(['25232910'])) &
               (df.xprod.str.contains('cimento|cp ii',regex=True)) &
               (df.xprod.str.contains('40 kg')) &
               (df.definicao.isnull()), 'definicao'] = 'cimento 40kg'
        # cimento 25kg
        df.loc[(df.ncm.isin(['25232910'])) &
               (df.xprod.str.contains('cimento|cp ii',regex=True)) &
               (df.xprod.str.contains('25 kg')) &
               (df.definicao.isnull()), 'definicao'] = 'cimento 25kg'
        # cal hidratado 5kg
        df.loc[(df.ncm.isin(['25223000','25221000','25222000'])) &
                    (df.xprod.str.contains('cal')) &
                    (df.xprod.str.contains('hid')) &
                    (df.xprod.str.contains('5 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'cal hidratado 5kg'


        """  ################## BORRACHA E SUAS OBRAS   ##################   """


        # luva cirurgica esteril
        df.loc[(df.ncm.isin(['40151100','40151900'])) &
                            (df.xprod.str.contains('luva')) &
                            (df.xprod.str.contains('cir')) &
                            (df.xprod.str.contains('est')) &
                            (df.definicao.isnull()), 'definicao'] = 'luva cirurgica esteril'
                    
        # pneu 275/80
        df.loc[(df.ncm.isin(['40112090','40111000'])) &
                            (df.xprod.str.contains('pneu')) &
                            (df.xprod.str.contains('275 80')) &
                            (df.definicao.isnull()), 'definicao'] = 'pneu 275/80'


        """  ################## AÇUCARES   ##################   """


        # açucar cristal
        df.loc[(df.ncm.isin(['17019900','02071400','34022000', '34022000', '15079011', '25232910'])) &  
                    (df.xprod.str.contains('acucar')) &
                    (df.xprod.str.contains('cristal|crital',regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'açucar cristal 1kg'
        # açucar demerara
        df.loc[(df.ncm.isin(['17039000'])) &  
                    (df.xprod.str.contains('acucar')) &
                    (df.xprod.str.contains('demerara')) &
                    (df.definicao.isnull()), 'definicao'] = 'açucar demerara 1kg'
        # açucar refinado 1kg
        df.loc[(df.ncm.isin(['17019900'])) & (
                        ((df.xprod.str.contains('acucar')) &
                        (df.xprod.str.contains('ref'))) |
                        ((df.xprod.str.contains('acucar')) &
                        (df.xprod.str.contains('branco')))) &
                        (df.xprod.str.contains('30 kg')) &
                        (df.definicao.isnull()), 'definicao'] = 'açucar refinado 30kg'
        # açucar refinado 1kg
        df.loc[(df.ncm.isin(['17019900', '19053100', '02071400','34022000', '34022000', '15079011', '25232910'])) & (                
                        ((df.xprod.str.contains('acucar')) &
                        (df.xprod.str.contains('ref'))) 
                        |
                        ((df.xprod.str.contains('acucar')) &
                        (df.xprod.str.contains('branco'))) 
                        |                
                        (df.xprod.str.contains('acucar')) ) &
                        (df.definicao.isnull()), 'definicao'] = 'açucar refinado 1kg'
        # rapadura 500g
        df.loc[(df.ncm.isin(['17039000','17031000'])) &  
                    (df.xprod.str.contains('500 g')) &
                    (df.xprod.str.contains('rapadura')) &
                    (df.definicao.isnull()), 'definicao'] = 'rapadura 500g'
        # achocolatado 400g
        df.loc[(df.ncm.isin(['18069000','18061000'])) &  
                    (df.xprod.str.contains('400 g')) &
                    (df.xprod.str.contains('achoc')) &
                    (df.definicao.isnull()), 'definicao'] = 'achocolatado 400g'
        # achocolatado 200g
        df.loc[(df.ncm.isin(['18069000','18061000'])) &  
                    (df.xprod.str.contains('200 g')) &
                    (df.xprod.str.contains('achoc')) &
                    (df.definicao.isnull()), 'definicao'] = 'achocolatado 200g'
        # achocolatado 300g
        df.loc[(df.ncm.isin(['18069000','18061000'])) &  
                    (df.xprod.str.contains('300 g')) &
                    (df.xprod.str.contains('achoc')) &
                    (df.definicao.isnull()), 'definicao'] = 'achocolatado 300g'
        # achocolatado 1kgg
        df.loc[(df.ncm.isin(['18069000','18061000'])) &  
                    (df.xprod.str.contains('1 kg')) &
                    (df.xprod.str.contains('achoc')) &
                    (df.definicao.isnull()), 'definicao'] = 'achocolatado 1kg'

        """  ################## POLVORA   ##################   """

        # fosforo
        df.loc[(df.ncm.isin(['36050000'])) &  
                    (df.xprod.str.contains('fosforo')) &
                    (df.definicao.isnull()), 'definicao'] = 'fosforo'


        """  ################## REFEIÇÃO   ##################   """


        # refeição
        # OBS: Não há distinção de café da manhã, almoço, jantar, refeicao ou se tem suco
        df.loc[(df.ncm.isin(['21069090', '21042000'])) &
               (df.xprod.str.contains('refeicao|janta|almoco|quentinha|cafe manha|coffee break|lanche|buffet',regex=True)) &
               (df.definicao.isnull()), 'definicao'] = 'refeição'


        """  ################## PAPEL   ##################   """


        # papel higienico c/4 rolos
        df.loc[(df.ncm.isin(['48181000'])) &  
                    (df.xprod.str.contains('papel')) &
                    (df.xprod.str.contains('c 4 ')) &
                    (df.xprod.str.contains('hig')) &
                    (df.definicao.isnull()), 'definicao'] = 'papel higienico c/4 rolos'
        # papel toalha c/2 rolos
        df.loc[(df.ncm.isin(['48181000'])) &  
                    (df.xprod.str.contains('papel')) &
                    (df.xprod.str.contains('c 2 '))&
                    (df.xprod.str.contains('toalha')) &
                    (df.definicao.isnull()), 'definicao'] = 'papel toalha c/2 rolos'
        # guardanapo 14x14cm
        df.loc[(df.ncm.isin(['48181000','48189090','48183000'])) &  
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('14 14')) &
                    (df.definicao.isnull()), 'definicao'] = 'guardanapo 14x14cm'
        # guardanapo 22x22cm
        df.loc[(df.ncm.isin(['48181000','48189090','48183000'])) &  
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('22 22')) &
                    (df.definicao.isnull()), 'definicao'] = 'guardanapo 22x22cm'
        # guardanapo 22x20cm
        df.loc[(df.ncm.isin(['48181000','48189090','48183000'])) &  
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('22 20')) &
                    (df.definicao.isnull()), 'definicao'] = 'guardanapo 22x20cm'
        # guardanapo 23x23cm
        df.loc[(df.ncm.isin(['48181000','48189090','48183000'])) &  
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('23 23')) &
                    (df.definicao.isnull()), 'definicao'] = 'guardanapo 23x23cm'
        # guardanapo 23x20cm
        df.loc[(df.ncm.isin(['48181000','48189090','48183000'])) &  
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('23 20')) &
                    (df.definicao.isnull()), 'definicao'] = 'guardanapo 23x20cm'


        """  ################## REMEDIOS   ##################   """


        # teste covid-19 
        df.loc[(df.ncm.isin(['30021229','30021590','30049099'])) & 
                            (df.xprod.str.contains('covid|corona',regex=True)) &
                            ( (df.xprod.str.contains('rapido')) |
                            (df.xprod.str.contains('teste')) |
                            (df.xprod.str.contains('igm')) |
                            (df.xprod.str.contains('igg'))
                            ) &
                        (df.definicao.isnull()), 'definicao'] = 'teste covid-19'
        # ivermectina 6mg
        df.loc[(df.ncm.isin(['30049059'])) &  
              (df.xprod.str.contains('ivermec')) &
              (df.definicao.isnull()), 'definicao'] = 'ivermectina 6mg'
        # azitromicina 40mg
        df.loc[(df.ncm.isin(['30042029','30032029','30042099','30049099','30041019','30049037','30049059'])) &  
                    (df.xprod.str.contains('azitromicina')) &
                    (df.xprod.str.contains('40 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'azitromicina 40mg'
        # azitromicina 200mg
        df.loc[(df.ncm.isin(['30042029','30032029','30042099','30049099','30041019','30049037','30049059'])) &  
                    (df.xprod.str.contains('azitromicina')) &
                    (df.xprod.str.contains('200 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'azitromicina 200mg'
        # azitromicina 500mg
        df.loc[(df.ncm.isin(['30042029','30032029','30042099','30049099','30041019','30049037','30049059'])) &  
                    (df.xprod.str.contains('azitromicina')) &
                    (df.xprod.str.contains('500 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'azitromicina 500mg'
        # azitromicina 600mg
        df.loc[(df.ncm.isin(['30042029','30032029','30042099','30049099','30041019','30049037','30049059'])) &  
                    (df.xprod.str.contains('azitromicina')) &
                    (df.xprod.str.contains('600 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'azitromicina 600mg'
        # azitromicina 900mg
        df.loc[(df.ncm.isin(['30042029','30032029','30042099','30049099','30041019','30049037','30049059'])) &  
                    (df.xprod.str.contains('azitromicina')) &
                    (df.xprod.str.contains('900 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'azitromicina 900mg'
        # hidroxicloroquina 400mg
        df.loc[(df.ncm.isin(['30049069'])) & 
                    (df.xprod.str.contains('hidroxicloroquina')) &
                    (df.xprod.str.contains('400 mg')) &
                    (df.definicao.isnull()), 'definicao'] = 'hidroxicloroquina 400mg'

        # atadura crepom 10cm
        df.loc[(df.ncm.isin(['30059090','30049099','30051090'])) &  
                    (df.xprod.str.contains('atadura')) &
                    (df.xprod.str.contains('crep')) &
                    (df.xprod.str.contains('10 cm')) &
                    (df.definicao.isnull()), 'definicao'] = 'atadura crepom 10cm'
        # atadura crepom 12cm
        df.loc[(df.ncm.isin(['30059090','30049099','30051090'])) &
                    (df.xprod.str.contains('atadura')) &
                    (df.xprod.str.contains('crep')) &
                    (df.xprod.str.contains('12 cm')) &
                    (df.definicao.isnull()), 'definicao'] = 'atadura crepom 12cm'
        # atadura crepom 15cm
        df.loc[(df.ncm.isin(['30059090','30049099','30051090'])) &  
                    (df.xprod.str.contains('atadura')) &
                    (df.xprod.str.contains('crep')) &
                    (df.xprod.str.contains('15 cm')) &
                    (df.definicao.isnull()), 'definicao'] = 'atadura crepom 15cm'
        # atadura crepom 20cm
        df.loc[(df.ncm.isin(['30059090','30049099','30051090'])) &  
                    (df.xprod.str.contains('atadura')) &
                    (df.xprod.str.contains('crep')) &
                    (df.xprod.str.contains('20 cm')) &
                    (df.definicao.isnull()), 'definicao'] = 'atadura crepom 20cm'
        # atadura crepom 30cm
        df.loc[(df.ncm.isin(['30059090','30049099','30051090'])) &
                    (df.xprod.str.contains('atadura')) &
                    (df.xprod.str.contains('crep')) &
                    (df.xprod.str.contains('30 cm')) &
                    (df.definicao.isnull()), 'definicao'] = 'atadura crepom 30cm'

        # acido acetilsalicilico 100mg
        df.loc[(df.ncm.isin(['30049024','30039071'])) &
                            (df.xprod.str.contains('acido')) &
                            (df.xprod.str.contains('acetil')) &
                            (df.xprod.str.contains('100 m')) &
                            (df.definicao.isnull()), 'definicao'] = 'acido acetilsalicilico 100mg'

        # algodao hidrofilo 500g
        df.loc[(df.ncm.isin(['30049069','30059090','30049099','30059019','30051010'])) &
                            (df.xprod.str.contains('algod')) &
                            (df.xprod.str.contains('hidro')) &
                            (df.xprod.str.contains('500 g')) &
                            (df.definicao.isnull()), 'definicao'] = 'algodao hidrofilo 500g'

        # algodao hidrofilo 250g
        df.loc[(df.ncm.isin(['30049069','30059090','30049099','30059019','30051010'])) &
                            (df.xprod.str.contains('algod')) &
                            (df.xprod.str.contains('hidro')) &
                            (df.xprod.str.contains('250 g')) &
                            (df.definicao.isnull()), 'definicao'] = 'algodao hidrofilo 250g'
        # soro 500ml
        df.loc[(df.ncm.isin(['30049099'])) &
                ( (df.xprod.str.contains('solucao|soro',regex=True)) |
                  ((df.xprod.str.contains('solucao|soro',regex=True)) &
                  (df.xprod.str.contains('fisiologic|cloreto de sodio',regex=True))) |
                  (df.xprod.str.contains('cloreto de sodio'))
                ) & ((df.xprod.str.contains('500')) | (~df.xprod.str.contains('ml'))) &
                (df.definicao.isnull()), 'definicao'] = 'soro fisiologico 500ml'
        # soro 250ml
        df.loc[(df.ncm.isin(['30049099'])) &
                ( (df.xprod.str.contains('solucao|soro',regex=True)) |
                  ((df.xprod.str.contains('solucao|soro',regex=True)) &
                  (df.xprod.str.contains('fisiologic|cloreto de sodio',regex=True))) |
                  (df.xprod.str.contains('cloreto de sodio',regex=True))
                ) & (df.xprod.str.contains('250')) &
                (df.definicao.isnull()), 'definicao'] = 'soro fisiologico 250ml'
        # soro 100ml
        df.loc[(df.ncm.isin(['30049099'])) &
                ( (df.xprod.str.contains('solucao|soro',regex=True)) |
                  ((df.xprod.str.contains('solucao|soro',regex=True)) &
                  (df.xprod.str.contains('fisiologic|cloreto de sodio',regex=True))) |
                  (df.xprod.str.contains('cloreto de sodio'))
                ) & (df.xprod.str.contains('100')) &
                (df.definicao.isnull()), 'definicao'] = 'soro fisiologico 100ml'

        """  ################## LEITE   ##################   """

       # REGRA DE NEGOCIO: Leite sem unidade considerado 350g
        # leite po int 200g 
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('200 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 200g'
        # leite po int 250g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) & 
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('250 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 250g'
        # leite po int 300g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('300 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 300g'
        # leite po int 400g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('400 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 400g'
        # leite po int 500g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('500 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 500g'
        # leite po int 800g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.xprod.str.contains('800 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 800g'
        # leite po 200g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('200 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 200g'
        # leite po 250g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('250 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 250g'
        # leite po 300g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('300 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 300g'
        # leite po 400g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('400 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 400g'
        # leite po 500g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('500 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 500g'
        # leite po 800g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('800 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 800g'
        # leite 1kg
        df.loc[(df.ncm.isin(['04022110'])) &  
               (df.xprod.str.contains('leite')) &
               (df.xprod.str.contains('po')) &
               (df.xprod.str.contains('1 kg')) &                
               (df.definicao.isnull()), 'definicao'] = 'leite em po 1kg'
        # leite po int 350g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.xprod.str.contains('int')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po integral 350g'
        # leite po 350g
        df.loc[(df.ncm.isin(['04022110'])) &  
                    (df.xprod.str.contains('leite')) &
                    (df.xprod.str.contains('po')) &
                    (df.definicao.isnull()), 'definicao'] = 'leite em po 350g'
        # leite 1lt
        df.loc[(df.ncm.isin(['04022110'])) &  
               (df.xprod.str.contains('leite')) &
               (df.xprod.str.contains('1 lt')) &
               (df.definicao.isnull()), 'definicao'] = 'leite 1lt'
        # bebida lactea 1lt
        df.loc[(df.ncm.isin(['04031000','04039000'])) &  
               (df.xprod.str.contains('beb')) &
               (df.xprod.str.contains('lac')) &
               (df.xprod.str.contains('1 lt')) & 
               (df.definicao.isnull()), 'definicao'] = 'bebida lactea 1lt'


        """  ################## LEGUMES   ##################   """


        # cenoura
        df.loc[(df.ncm.isin(['20060000', '07061000'])) &  
                    (df.xprod.str.contains('cenoura')) &
                    (df.definicao.isnull()), 'definicao'] = 'cenoura'
        # tomate
        df.loc[(df.ncm.isin(['20060000', '07020000', '07061000'])) & 
                    (df.xprod.str.contains('tomate')) &
                    (~df.xprod.str.contains('molho')) &
                    (df.definicao.isnull()), 'definicao'] = 'tomate'


        """  ################## FRUTAS E DERIVADOS   ##################   """

        # polpa de fruta 200g
        df.loc[(df.ncm.isin(['20089900','20083000','20099000','20060000'])) &  
               (df.xprod.str.contains('polpa')) &
               (df.xprod.str.contains('200 g')) &
               (df.definicao.isnull()), 'definicao'] = 'polpa de fruta 200g'
        # polpa de fruta 400g
        df.loc[(df.ncm.isin(['20089900','20083000','20099000','20060000'])) &  
               (df.xprod.str.contains('polpa')) &
               (df.xprod.str.contains('400 g')) &
               (df.definicao.isnull()), 'definicao'] = 'polpa de fruta 400g'
        # polpa de fruta 1kg
        df.loc[(df.ncm.isin(['20089900','20083000','20099000','20060000'])) &  
               (df.xprod.str.contains('polpa')) &
               ((~df.xprod.str.contains('kg| g | g$',regex=True)) |
               (df.ucom.str.contains('1 kg'))) &
               (df.definicao.isnull()), 'definicao'] = 'polpa de fruta 1kg'


        """  ################## CAFE   ##################   """

        # cafe 250g
        df.loc[(df.ncm.isin(['09012100'])) &  
                    (df.xprod.str.contains('caf')) &
                    (df.xprod.str.contains('250')) &
                    (df.definicao.isnull()), 'definicao'] = 'cafe 250g'
        # cafe 500g
        df.loc[(df.ncm.isin(['09012100'])) &  
                    (df.xprod.str.contains('caf')) &
                    (df.xprod.str.contains('500')) &
                    (df.definicao.isnull()), 'definicao'] = 'cafe 500g'


        """  ################## CEREAIS   ##################   """

        # arroz 1kg
        df.loc[(df.ncm.isin(['10063011','10062010','10063021'])) &  
                    (df.xprod.str.contains('arroz')) &
                    (~df.xprod.str.contains('cesta|porcao|farinha|mingau', regex=True)) &
                    (df.definicao.isnull()), 'definicao'] = 'arroz 1kg'
        # aveia em flocos 200g
        df.loc[(df.ncm.isin(['11041200'])) &  
                    (df.xprod.str.contains('aveia')) &
                    (df.xprod.str.contains('floc')) &
                    (df.xprod.str.contains('200 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'aveia em flocos 200g'  
        # amido de milho 500g
        df.loc[(df.ncm.isin(['11081200','11041200'])) &  
                    (df.xprod.str.contains('amid')) &
                    (df.xprod.str.contains('milho')) &
                    (df.xprod.str.contains('500 g')) &
                    (df.definicao.isnull()), 'definicao'] = 'amido de milho 500g'
        # farinha de milho flocada 500g 
        df.loc[(df.ncm.isin(['11041900','11022000','11042300'])) &  
                    (~df.xprod.str.contains('aveia|fuba|mugunza',regex=True)) &
                    ((df.xprod.str.contains('fari')) &
                    (df.xprod.str.contains('floc|cuscuz|milho',regex=True))) 
                    |
                    ( (df.xprod.str.contains('floc',regex=True)) &
                        (df.xprod.str.contains('milho')) ) &
                    (df.xprod.str.contains('500')) &
                    (df.definicao.isnull()), 'definicao'] = 'farinha de milho flocada 500g'        
        # goma de mandioca 1kg
        df.loc[(df.ncm.isin(['11062000','11081400', '11022000'])) &  
               (df.xprod.str.contains('goma|fecula',regex=True)) &
               (df.xprod.str.contains('mand')) &
               (df.definicao.isnull()), 'definicao'] = 'goma de mandioca 1kg'
        # farinha de mandioca 1kg
        df.loc[(df.ncm.isin(['11062000','11081400', '11022000'])) &  
                    (df.xprod.str.contains('far')) &
                    (df.xprod.str.contains('mand')) &
                    (df.definicao.isnull()), 'definicao'] = 'farinha de mandioca 1kg'
        # farinha de trigo 1kg
        df.loc[(df.ncm.isin(['11010010','11022000', '11062000'])) &  
                    (df.xprod.str.contains('far')) &
                    (df.xprod.str.contains('trig')) &
                    (df.xprod.str.contains('50 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'farinha de trigo 50kg'
        # farinha de trigo 10kg
        df.loc[(df.ncm.isin(['11010010','11022000', '11062000'])) &  
                    (df.xprod.str.contains('far')) &
                    (df.xprod.str.contains('trig')) &
                    (df.xprod.str.contains('10 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'farinha de trigo 10kg'
        # farinha de trigo 1kg
        df.loc[(df.ncm.isin(['11010010','11022000','11062000'])) &  
                    (df.xprod.str.contains('far')) &
                    (df.xprod.str.contains('trig')) &
                    (df.definicao.isnull()), 'definicao'] = 'farinha de trigo 1kg'

        """  ################## GORDURAS E OLEOS   ##################   """
        
        
        #margarina 1 kg
        df.loc[(df.ncm.isin(['15171000'])) &  
                    (df.xprod.str.contains('marg')) &
                    (df.xprod.str.contains('1 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'margarina 1 kg' 
        #margarina 3 kg
        df.loc[(df.ncm.isin(['15171000'])) &  
                    (df.xprod.str.contains('marg')) &
                    (df.xprod.str.contains('3 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'margarina 3 kg' 
        #margarina 15 kg
        df.loc[(df.ncm.isin(['15171000'])) &  
                    (df.xprod.str.contains('marg')) &
                    (df.xprod.str.contains('15 kg')) &
                    (df.definicao.isnull()), 'definicao'] = 'margarina 15 kg'
        #margarina 500g
        df.loc[(df.ncm.isin(['15171000'])) &  
                    (df.xprod.str.contains('marg')) &
                    ((~df.xprod.str.contains('kg')) |
                        (df.ucom.str.contains('kg')))&
                    (df.definicao.isnull()), 'definicao'] = 'margarina 500g' 

        # oleo soja 900ml
        df.loc[(df.ncm.isin(['15079011'])) &  
                    (df.xprod.str.contains('oleo')) &
                    (df.xprod.str.contains('900 ml')) &
                    (df.xprod.str.contains('soja')) &
                    (df.definicao.isnull()), 'definicao'] = 'oleo de soja 900ml'
        # azeite 500ml
        df.loc[(df.ncm.isin(['15091000'])) &  
                    (df.xprod.str.contains('azeite')) &
                    (df.xprod.str.contains('500 ml')) &
                    (df.definicao.isnull()), 'definicao'] = 'azeite 500ml'
        
        
        """  ################## ALEATÓRIOS   ##################   """
        
        # filtro de oleo
        # Não leva em consideração o tipo de filtro
        df.loc[(df.ncm.isin(['84212300'])) &  
                    (df.xprod.str.contains('filtr')) &
                    (df.xprod.str.contains('oleo')) &
                    (df.definicao.isnull()), 'definicao'] = 'filtro de oleo'
        # filtro de combustivel
        # Não leva em consideração o tipo de filtro
        df.loc[(df.ncm.isin(['84212300'])) &  
                    (df.xprod.str.contains('filtr')) &
                    (df.xprod.str.contains('combust')) &
                    (df.definicao.isnull()), 'definicao'] = 'filtro de combustivel'
        # unidade de evaporadora split 12k
        df.loc[(df.ncm.isin(['84151011'])) &  
                    (df.xprod.str.contains('split')) &
                    (df.xprod.str.contains('12 k|12000',regex=True)) &
                    (df.xprod.str.contains('evap')) &
                    (df.definicao.isnull()), 'definicao'] = 'unidade de evaporadora split 12k'
        # unidade de evaporadora split 18k
        df.loc[(df.ncm.isin(['84151011'])) &  
                    (df.xprod.str.contains('split')) &
                    (df.xprod.str.contains('18 k|18000',regex=True)) &
                    (df.xprod.str.contains('evap')) &
                    (df.definicao.isnull()), 'definicao'] = 'unidade de evaporadora split 18k'
        #cesta basica
        df.loc[(df.ncm.isin(['21069090'])) &  
                    (df.xprod.str.contains('cesta basica')) &
                    (df.definicao.isnull()), 'definicao'] = 'cesta basica'
        #balde plastico 20lt
        df.loc[(df.ncm.isin(['39241000','28289011'])) &
               (df.xprod.str.contains('balde|lixeira|bacia|cesto', regex=True)) &
               (df.xprod.str.contains('20')) &
               (df.definicao.isnull()), 'definicao'] = 'balde plastico 20lt'
       
        
        """  ################## OUTROS   ##################   """
        
        df.loc[df.definicao.isnull(), 'definicao'] = 'outros'
        
        return df

    @staticmethod
    def redefine_ucom(df):
        '''
        Redefine Unidade de Comercialização de acordo com padrão para ser visualizado, evitando redundância de unidade.
		Args:
			df(pandas.DataFrame): Contendo pelo menos uma coluna 'xprod' e 'ucom' do tipo string
		Retorna:
			df(pandas.DataFrame): Retorna o mesmo objeto recebido, com coluna 'ucom' alterada.
        '''
        
        """  ################## COMBUSTIVEIS   ##################   """

        df.loc[(df.xprod.str.contains('diesel|gasolina|etanol',regex=True)), 'ucom' ]= 'lt'
        df.loc[(df.xprod.str.contains('glp| gas ', regex=True)), 'ucom' ] = 'und'

        """  ################## CARNES E FRANGOS   ##################   """

        df.loc[( (df.xprod.str.contains('pto|peito', regex=True)) &
                 (df.xprod.str.contains('frango|fgo',regex=True))  
                ) |
                (df.xprod.str.contains('coxa'))
                |
                (df.xprod.str.contains('frango|fgo|carne'))
                |
                (
                (df.xprod.str.contains('costela')) &
                (df.xprod.str.contains('bov'))
                ), 'ucom'] = 'kg'

        """  ################## BISCOITOS E MACARRAO   ##################   """

        df.loc[( (
                (df.xprod.str.contains('cream|crem', regex=True)) &
                (df.xprod.str.contains('crack|crak', regex=True)) 
                ) |
                (
                (df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('sal'))
                ) |
                (df.xprod.str.contains('maizena|maisena|rosquinha|maria',regex=True))
                  |
                (
                (df.xprod.str.contains('mac')) &
                (df.xprod.str.contains('espag|spaghet', regex=True)) 
                ) |
                (
                (df.xprod.str.contains('mac')) &
                (df.xprod.str.contains('integral', regex=True)) 
                ) ) & (df.ucom.isin(['kg','pct'])) , 'ucom']= 'und'
        df.loc[( (
                (df.xprod.str.contains('cream|crem', regex=True)) &
                (df.xprod.str.contains('crack|crak', regex=True)) 
                ) |
                (
                (df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('sal'))
                ) |
                (df.xprod.str.contains('maizena|maisena|rosquinha|maria',regex=True))
                | (
                (df.xprod.str.contains('mac')) &
                (df.xprod.str.contains('espag|spaghet', regex=True)) 
                ) |
                (
                (df.xprod.str.contains('mac')) &
                (df.xprod.str.contains('integral', regex=True)) 
                ) ) & (df.ucom.isin(['frd'])), 'ucom' ]= 'cx'

        """  ################## AGUA MIN E REFRI   ##################   """

        df.loc[( (
                (df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('(500|510|350|300|200|1.5|1,5) (ml|lt)',regex=True))
                ) |
                (
                ((df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('20 lt'))) |
                ((df.xprod.str.contains('garraf')) &
                (df.xprod.str.contains('20')))
                ) |
                (
                (df.xprod.str.contains('refr')) &
                (df.xprod.str.contains('2 lt'))
                ) ) & (~df.ucom.isin(['cx','frd','pct'])), 'ucom' ]= 'und'
        df.loc[( (
                (df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('(500|510|350|300|200|1.5|1,5) (ml|lt)',regex=True))
                ) |
                (
                ((df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('20 lt'))) |
                ((df.xprod.str.contains('garraf')) &
                (df.xprod.str.contains('20')))
                ) |
                (
                (df.xprod.str.contains('refr')) &
                (df.xprod.str.contains('2 lt'))
                ) )  & (~df.ucom.isin(['und'])), 'ucom' ]= 'frd'

        """  ################## DESCARTAVEIS   ##################   """

        df.loc[(df.xprod.str.contains('^copo', regex=True)) &
                (df.xprod.str.contains('( 50|150|180|200|250|300) ml',regex=True)) &
                (df.ucom.isin(['und'])), 'ucom' ]= 'pct'

        """  ################## LIMPEZA  E ALCOOL ##################   """

        df.loc[( (
                ((df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('sanit'))) |
                (df.xprod.str.contains('hipoclorito de sodio'))
               ) |
               (
                (df.xprod.str.contains('deterg|det |sabao liq|sabao|desinf|limpa|limp |veja', regex=True))
               ) ) & (df.ucom.isin(['pct','frd'])), 'ucom' ]= 'cx'
        df.loc[( (
                ((df.xprod.str.contains('agua')) &
                (df.xprod.str.contains('sanit'))) |
                (df.xprod.str.contains('hipoclorito de sodio'))
               ) |
               (
                (df.xprod.str.contains('deterg|det |sabao liq|sabao|desinf|limpa|limp |veja', regex=True))
               ) |
               (
                (df.xprod.str.contains('alcool')) &
                (df.xprod.str.contains('gel|etilico|92|96|46|54|64|99',regex=True))
               ) ) & (~df.ucom.isin(['cx'])), 'ucom' ] = 'und'

        """  ##################  SAL   ##################   """

        df.loc[( ((df.xprod.str.contains('sal ')) &
                 (df.xprod.str.contains('ref|1 kg|iod',regex=True))                    
                ) | (df.xprod.str.contains('^sal$|^sal ', regex=True)) ) 
                & (df.ucom.isin(['pct','kg'])), 'ucom' ] = 'und'        
        
        """  ################## CONSTRUÇÃO   ##################   """

        df.loc[( (df.xprod.str.contains('cimento|cp ii',regex=True)) |
                (
                (df.xprod.str.contains('cal')) &
                (df.xprod.str.contains('hid')) 
                ) ) & (df.ucom.isin(['kg','und','pct'])), 'ucom' ] = 'sc'

        """  ################## BORRACHA E SUAS OBRAS   ##################   """

        df.loc[(df.xprod.str.contains('luva')) &
                (df.xprod.str.contains('cir')) &
                (df.xprod.str.contains('est')) &
                (df.ucom.isin(['pct'])), 'ucom' ] = 'und'                            

        """  ################## AÇUCARES   ##################   """

        df.loc[( ((df.xprod.str.contains('acucar')) &
                  (df.xprod.str.contains('cristal|crital|ref|branco|demerara',regex=True))) 
                | (
                   (df.xprod.str.contains('acucar')) &
                    (df.xprod.str.contains('kg'))
                ) |
                (df.xprod.str.contains('rapadura')) 
                | (df.xprod.str.contains('achoc')) )
                 & (df.ucom.isin(['kg'])), 'ucom' ] = 'und'
        df.loc[( (df.xprod.str.contains('acucar')) &
                (df.xprod.str.contains('cristal|crital|ref|branco|demerara',regex=True)) 
                |(
                   (df.xprod.str.contains('acucar')) &
                    (df.xprod.str.contains('1 kg'))
                ) |
                (df.xprod.str.contains('rapadura'))
                | (df.xprod.str.contains('achoc')) )
                & (df.ucom.isin(['cx'])), 'ucom' ] = 'frd'
        
        """  ################## POLVORA   ##################   """

        df.loc[(df.xprod.str.contains('fosforo')) &
                (~df.ucom.isin(['und'])), 'ucom' ] = 'pct'
                
        """  ################## PAPEL   ##################   """

        df.loc[( (df.xprod.str.contains('papel')) &
                (df.xprod.str.contains('c (4|2) ',regex=True)) &
                (df.xprod.str.contains('hig')) 
                | (
                    (df.xprod.str.contains('guardanapo')) &
                    (df.xprod.str.contains('14 14|22 22|22 20|23 23|23 20',regex=True)) 
                ) ) & (~df.ucom.isin(['cx'])), 'ucom' ] = 'und'
 
        """  ################## REMEDIOS   ##################   """


        df.loc[ (df.xprod.str.contains('covid|corona',regex=True)) &
                (df.xprod.str.contains('rapido|teste|igm|igg|ivermec|azitromicina|hidroxicloroquina',regex=True)) |
                ( (df.xprod.str.contains('solucao|soro',regex=True)) |
                  ((df.xprod.str.contains('solucao|soro',regex=True)) &
                  (df.xprod.str.contains('fisiologic|cloreto de sodio',regex=True))) |
                  (df.xprod.str.contains('cloreto de sodio'))
                ) |
                (
                (df.xprod.str.contains('acido')) &
                (df.xprod.str.contains('acetil'))
                ) |
                (
                (df.xprod.str.contains('algod')) &
                (df.xprod.str.contains('hidro'))
                ) |
                (
                  (df.xprod.str.contains('solucao|soro',regex=True)) |
                  ((df.xprod.str.contains('solucao|soro',regex=True)) &
                  (df.xprod.str.contains('fisiologic|cloreto de sodio',regex=True))) |
                  (df.xprod.str.contains('cloreto de sodio'))
                ), 'ucom'] = 'und'
        df.loc[ (df.xprod.str.contains('atadura')) &
                (df.xprod.str.contains('crep')) & 
                (~df.ucom.isin(['pct'])), 'ucom' ] = 'und'

        """  ################## LEITE   ##################   """

        df.loc[( ((df.xprod.str.contains('leite')) &
                 (df.xprod.str.contains('po|lt ',regex=True)) )
                | (
                (df.xprod.str.contains('beb')) &
                (df.xprod.str.contains('lac'))
                ) ) & (~df.ucom.isin(['und'])), 'ucom' ] = 'und'

        """  ################## LEGUMES   ##################   """

        df.loc[(df.xprod.str.contains('cenoura|tomate',regex=True)) &
                (~df.xprod.str.contains('molho')) &
                (~df.ucom.isin(['kg'])), 'ucom' ] = 'kg'

        """  ################## FRUTAS E DERIVADOS   ##################   """
        
        df.loc[(df.xprod.str.contains('polpa')) &
               (df.xprod.str.contains('kg| g | g$',regex=True)) &
               (df.ucom.isin(['kg','pct'])), 'ucom' ] = 'und' 

        """  ################## CAFE   ##################   """

        df.loc[(df.xprod.str.contains('caf')) &
                (df.xprod.str.contains('kg| g | g$',regex=True)) &
                (df.ucom.isin(['kg','pct'])), 'ucom' ] = 'und'
        
        df.loc[(df.xprod.str.contains('caf')) &
                (df.xprod.str.contains('kg| g | g$',regex=True)) &
                (df.ucom.isin(['cx'])), 'ucom' ] = 'frd'

        """  ################## CEREAIS   ##################   """

        df.loc[( (
                (df.xprod.str.contains('arroz')) &
                (~df.xprod.str.contains('cesta|porcao|farinha|mingau', regex=True))
                ) |
                (   (df.xprod.str.contains('aveia')) &
                    (df.xprod.str.contains('floc'))) 
                | ( (df.xprod.str.contains('amid')) &
                    (df.xprod.str.contains('milho')))
                | ( (df.xprod.str.contains('fari')) &
                    (df.xprod.str.contains('floc|cuscuz|milho|mand|trig',regex=True))) 
                |( (df.xprod.str.contains('floc')) &
                    (df.xprod.str.contains('milho')) ) |    
               (df.xprod.str.contains('goma|fecula',regex=True)) )
               & (~df.ucom.isin(['frd'])), 'ucom' ] = 'und'

        """  ################## GORDURAS E OLEOS   ##################   """
        
        df.loc[( (
                (df.xprod.str.contains('marg')) &
                (df.xprod.str.contains('kg')) 
                ) |
                (
                (df.xprod.str.contains('oleo')) &
                (df.xprod.str.contains('soja'))
                ) |
                (df.xprod.str.contains('azeite')) ) 
                & (~df.ucom.isin(['cx'])), 'ucom' ] = 'und'

        """  ################## ALEATÓRIOS   ##################   """
        
        df.loc[( (
                (df.xprod.str.contains('filtr')) &
                (df.xprod.str.contains('oleo'))
                ) |
                (
                (df.xprod.str.contains('filtr')) &
                (df.xprod.str.contains('combust'))
                ) |
                (
                (df.xprod.str.contains('split')) &
                (df.xprod.str.contains('12 k|12000|18 k|18000',regex=True)) &
                (df.xprod.str.contains('evap'))
                ) | (df.xprod.str.contains('cesta basica')) |
                (
                (df.xprod.str.contains('balde|lixeira|bacia|cesto', regex=True)) &
                (df.xprod.str.contains('20'))) )
                & (~df.ucom.isin(['und'])), 'ucom' ] = 'und'
        
        return df


    def calc_preco(self, df):
        '''
        Calcula preco médio, minimo e máximo para a coluna de precos 'Valor Unitario de Comercializacao' (vuncom) para cada classe presente na coluna 'classe' e
		salva um arquivo csv com os preços para cada classe em formato brasileiro de float (usado em visualização).

		Args:
			df(pd.DataFrame): dataframe contendo produtos com coluna de preços 'Valor Unitario de Comercializacao' (vuncom) e 'classe' (determinada na etapa de predição)
        '''
        le_filename = self.config_files['label_encoder_filename']
        le = self.abre(self.path_files, f'{le_filename}.pkl')
        descricoes = le.classes_
        grupos = le.transform(le.classes_)

        groupos_id = []
        group_names = []
        media = []
        mediana = []
        media_saneada = []
        precos_max = []
        precos_min = []

        for i, grupo in enumerate(grupos):
            try:
                df_grupo = df[df['classe'] == grupo]

                #S = E – M sobrepreço estimado - referencial mercado
                media.append(round(df_grupo['vuncom'].mean(),3))
                mediana.append(round(df_grupo['vuncom'].median(),3))
                nova_media, preco_max, preco_min = self.calc_media_saneada(df_grupo)
                
                media_saneada.append(nova_media)
                precos_max.append(preco_max)
                precos_min.append(preco_min)

                descricao = descricoes[i]
                group_names.append(descricao)
                
                groupos_id.append(i+1)
            except:
                pass
        df_precos = pd.DataFrame(zip(groupos_id, group_names, media, mediana, precos_max, precos_min, media_saneada),\
                                    columns=["id","Classe", "Media", "Mediana", "Max", "Min", "Media Saneada"])

        df_precos = df_precos.round(2)
        df_precos['Media'] = df_precos['Media'].astype(str).str.replace('.',",")  
        df_precos['Mediana'] = df_precos['Mediana'].astype(str).str.replace('.',",")  
        df_precos['Max'] = df_precos['Max'].astype(str).str.replace('.',",")  
        df_precos['Min'] = df_precos['Min'].astype(str).str.replace('.',",")  
        df_precos['Media Saneada'] = df_precos['Media Saneada'].astype(str).str.replace('.',",")

        df_precos.set_index('id', inplace=True)
        df_precos.to_csv(''.join((self.path_csvs, f'{self.precos_filename}.csv')),encoding='utf-8-sig')

    
    def filtra_produtos(self, df, method='Predict'):
        '''
        Filtra produtos por ncms selecionados em caso de treinamento, salva descrição original para uso em visualização e descosidera registros com descrições vazias.
		Args:
			df(pandas.DataFrame): Registros de produtos com colunas 'vuncom', 'xprod' e 'ncm'.
            method(str)         : Modificador de filtragem a partir do uso, 'Treino' ou 'Predict' (default).
		Retorna:
			df(pandas.DataFrame): Quantidade de registros filtrados.
        '''
        df['ncm'] = df.ncm.astype(str)
        df['ncm'] = df['ncm'].apply(lambda x: ''.join(['0',x]) if len(x) == 7 else x)
        df['vuncom'] = df['vuncom'].replace('^\s*$|^$', 'und', regex=True)
        df['Produto'] = df['xprod']
        if method == 'Train':
            df = df[df['ncm'].isin(self.lista_ncms)]

        return df


    def limpa_ucom(self, df):
        '''
        Padroniza unidades comercializadas da coluna 'ucom' para unidades padrões selecionadas.
        Facilita filtro em visualização, logo usado apenas na etapa de predição.
		Args:
			df(pandas.DataFrame): Contendo coluna 'ucom' do tipo string
		Retorna:
			df(pandas.DataFrame): Mesmo objeto com coluna 'ucom' alterada.
        '''
        df.loc[df.ucom.isin(self.lista_lt), 'ucom'] = 'lt'
        df.loc[df.ucom.isin(self.lista_kg), 'ucom'] = 'kg'
        df.loc[df.ucom.isin(self.lista_und), 'ucom'] = 'und'
        df.loc[df.ucom.isin(self.lista_pct), 'ucom'] = 'pct'
        df.loc[df.ucom.isin(self.lista_cx), 'ucom'] = 'cx'
        df.loc[df.ucom.isin(self.lista_frd), 'ucom'] = 'frd'
        df.loc[df.ucom.isin(self.lista_saco), 'ucom'] = 'sc'
        df.loc[~df.ucom.isin(self.unidades), 'ucom'] = 'und' 
        
        return df


    def limpa_descricao(self, df):
        '''
        Realiza métodos de limpeza padrão em todas as descrição.
		Args:
			df(pandas.DataFrame): Contendo pelo menos uma coluna 'xprod' do tipo string
		Retorna:
			df(pandas.DataFrame): Retorna o mesmo objeto recebido, com alterações de texto na coluna 'xprod'
        '''
        def remove_numeros_inicio(descr):
            while ( (len(descr) > 1) and (descr.split()[0].isdigit()) ):
                descr = ' '.join( descr.split()[1:] )
            return descr

        #limpa sentencas retirando stopwords, pontuacao e deixa minusculo.
        df['xprod'] = [ ' '.join([word.lower() for word in descr.split() if word.lower() not in self.pt_stopwords]) for descr in df['xprod'].astype(str)]
        #insere espaco apos / e -, pra no final nao ficar palavras assim: csolucao, ptexto (originais eram c/solucao, p-texto)
        df['xprod'] = df['xprod'].apply(lambda descr: re.sub(r'/|-',r' ',descr))
        # retira . entre palavras tipo det.liqu 
        df['xprod'] = df['xprod'].apply(lambda x: ' '.join( [re.sub(r'\.',r' ', word) if '.' in word and not any(char.isdigit() for char in word) else word for word in x.split()] ) )
        #retira pontuacao (com exceção de virgula e ponto dos numeros):
        df['xprod'] = df['xprod'].apply(lambda x: ''.join(re.findall('\d{1}\.\d{1}|\d{1}\,\d{1}|\w|\s',x)))
        #insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'(\d+)([A-Za-z])',r'\1 \2',x))
        # insere espaco apos letra e numero ex.:c100 pc50
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'([A-Za-z]{1})(\d+)',r'\1 \2',x))
        # insere espaco se unidade esta colada em outro medida ex.: 5litros 
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'(\d+)(l|lt|litro|litros|li|kg|ml|mg|cm|g|gr|kga)',r'\1 \2',x))
        #apaga caracteres pequenos se n estiver nas condições
        df['xprod'] = df['xprod'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1 or word == 'c' or word == 's' or word in self.unis or word.isdigit()]))
        #retira espacos duplicados
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r' +',r' ', x))
        #retira espaco no inicio da frase
        df['xprod'] = df['xprod'].apply(lambda x: x.strip())
        #retira acentos:
        df['xprod'] = df['xprod'].apply(lambda x: unidecode.unidecode(x))
        # remove zeros a esquerda de numeros (02 litros, 05, etc.)
        df['xprod'] = df['xprod'].apply(lambda x: ' '.join([word.lstrip('0') for word in x.split()] ) )
        #remove registros com descricao vazia
        df['xprod'] = df['xprod'].replace('^\s*$|^$', np.nan, regex=True)
        df = df.dropna(subset=['xprod'])
        #muda l e lts e litros por lt
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r' lts | litros | litro | l | li ', r' lt ', x))
        #muda l e lts por lt se for no final de frase
        df['xprod'] = df['xprod'].apply(lambda x: x  if x.split()[-1] not in ['l','li','lts','litro','litros','ltrs'] else ' '.join((' '.join(x.split()[:-1]), 'lt')))
        #substitui 1000ml por 1l 
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'1000 ml',r'1 lt',x))
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'5000 ml',r'5 lt',x))
        #limpa palavras novamente
        df['xprod'] = [ ' '.join([word for word in descr.split() if word.lower() not in self.pt_stopwords]) for descr in df['xprod'].astype(str)]
        #se primeira palavra for numero apaga (verifica varias vezes p/ casos com mts numeros)      
        df['xprod'] = df['xprod'].apply(lambda x: remove_numeros_inicio(x))
        # apaga 40.00 kg
        df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'(\d+\.\d+) (kg|un)',r' ',x))

        return df


    def preprocessing(self, df, class_size=800, save=True):
        '''
        Realiza codificações de classe e descrições utilizando countvectorizer, labelencoder e tf-idf; divide registros para teste e treino e aplicar método de resampling
        para aplicar no modelo de classificação. Por fim salva objetos treinados sobre descrições de treinamento.
		Args:
			df(pandas.DataFrame): Registros total de produtos que serão filtrados e preprocessados para estarem apteis a irem para o modelo
            class_size(int)     : Valor mínimo para que produtos de uma classe sejam considerados pelo classificador.
            save(bool)          : Determinar se deve ou não salvar arquivos .pkl dos codificadores (usados na predição)
		Retorna:
			df_aux(pd.dataframe)        : Contendo amostras que não serão usadas para treinamento pois entraram na classe 'outros'.
            X_train_tfidf(sparse matrix): Amostras codificadas para treinamento
            X_test_tfidf(sparse matrix) : Amostras codificadas para teste
            y_train(list)               : Classes codificadas para treinamento
            y_test(list)                : Classes codificadas para teste
        '''        

        df_aux = df[df['definicao'].isin(['outros'])]
        df = df[~df['definicao'].isin(['outros'])]
        
        count_vect = CountVectorizer(ngram_range=(2,4),max_features=7000, min_df=2)
        tfidf_transformer = TfidfTransformer()
        le = LabelEncoder()
        
        df.definicao = le.fit_transform(df.definicao)
        counts = df['definicao'].value_counts()
        df = df[df['definicao'].isin(counts[counts > class_size].index)]
        df = df.loc[~df.definicao.isin(le.transform(['filtro de oleo', 'detergente 500ml', 'soro fisiologico 500ml',\
                                    'copo descartavel 150ml','leite em po integral 200g']).tolist())]
        
        X_train, X_test, y_train, y_test = train_test_split(df['xprod'], df['definicao'], random_state= 0)
        
        X_train_counts = count_vect.fit_transform(X_train)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        
        X_test_counts = count_vect.transform(X_test)
        X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

        X_train_tfidf, y_train = self.resampling(X_train_tfidf, y_train)
        
        if save:
            count_vect_filename = self.config_files['count_vect_filename']
            self.grava(count_vect, self.path_files, f'{count_vect_filename}.pkl')
            tfidf_filename = self.config_files['tfidf_transformer_filename']
            self.grava(tfidf_transformer, self.path_files, f'{tfidf_filename}.pkl')
            encoder_filename = self.config_files['label_encoder_filename']
            self.grava(le, self.path_files, f'{encoder_filename}.pkl')

        return df_aux, X_train_tfidf, X_test_tfidf, y_train, y_test
    

    def treinar_modelo(self, X_train_tfidf, X_test_tfidf, y_train, y_test, save=True):
        '''
        Treina modelo MultinomialNB com cross validation e salva classes aprendidas assim como modelo treinado.
		Args:
			X_train_tfidf(sparse matrix): Amostras codificadas para treinamento
            X_test_tfidf(sparse matrix) : Amostras codificadas para teste
            y_train(list)               : Classes codificadas para treinamento
            y_test(list)                : Classes codificadas para teste
            save(bool)                  : Se deve salvar objetos .pkl contendo classes aprendidas e o modelo treinado, ou não (necessários para predição).
        '''
        clf = MultinomialNB()
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X_train_tfidf)        

        for train_index, test_index in kf.split(X_train_tfidf, y_train):
            X_train, _ = X_train_tfidf[train_index], X_train_tfidf[test_index]
            Y_train, _ = y_train[train_index], y_train[test_index]
            clf = clf.fit(X_train, Y_train)
        
        y_pred = clf.predict(X_test_tfidf)
        print(f'Acurácia: {accuracy_score(y_test,y_pred)}')
        
        if save:
            classes = self.config_files['classes_treinamento']
            clf_filename = self.config_files['clf_filename']
            self.grava(set(y_train), self.path_files, f'{classes}.pkl')
            self.grava(clf, self.path_files, f'{clf_filename}.pkl')


    def fazer_predicao_probabilistica(self, df, threshold=0.9):
        '''
        Realiza predição baseado no threshold probabilístico fornecido e classifica produtos a partir de modelo treinado.
		Args:
			df(pandas.DataFrame): Contendo registros a serem classificados pelo modelo treinado (predição)
            threshold(number)   : Delimitador de certeza do modelo se descrição pertence a uma certa classe.
		Retorna:
			df(pandas.DataFrame): Retorna o mesmo objeto, com coluna nova 'classe' contendo apenas classes preditas pelo modelo (ignorando as nulas).
        '''
        clf_filename = self.config_files['clf_filename']
        count_vect_filename = self.config_files['count_vect_filename']
        tfidf_filename = self.config_files['tfidf_transformer_filename']
        classes_treinamento = self.config_files['classes_treinamento']
        
        clf = self.abre(self.path_files, f'{clf_filename}.pkl')
        count_vect = self.abre(self.path_files, f'{count_vect_filename}.pkl')
        tfidf_transformer = self.abre(self.path_files, f'{tfidf_filename}.pkl')
        classes = self.abre(self.path_files, f'{classes_treinamento}.pkl')

        real_groups = sorted(classes)
        dictionary = dict(zip(list(range(len(real_groups))), real_groups))
        
        X_test_counts_others = count_vect.transform(df['xprod'])
        X_test_tfidf_others = tfidf_transformer.transform(X_test_counts_others)
        
        y_prob = clf.predict_proba(X_test_tfidf_others)
        y_pred_prob = [ dictionary[np.argmax(probs)] if max(probs) > threshold else None for probs in y_prob ]
        df['classe'] = y_pred_prob
        
        return df[~df['classe'].isna()]


    def exec_fluxo_treinamento(self, data='csv'):
        '''
        Aplica pipeline de treinamento completo ao receber descrições novas.  
		Args:
            data(str)               : Determina se o treinamento deve ser feito com dados de um csv ou se deve-se usar todos do banco.
        Retorna:
			df_aux(pandas.DataFrame): Retorna registros que não foram usados para treinamento pois obtiveram descrição 'outros' na pré-classificação. Registros salvos apenas para realizar testes de predição do modelo.
        '''
        if data != 'csv':
            df = self.df_produto
        else:
            produtos_csv = self.config_files['produtos_banco_filename']
            df = pd.read_csv(f'{os.path.join(self.path_files, produtos_csv)}.csv')

        df = self.filtra_produtos(df, method='Train')
        df = self.limpa_descricao(df)
        df = self.classifica_descricoes(df)
        df = self.metodos_pln(df)
        df_aux, X_train_tfidf, X_test_tfidf, y_train,y_test = self.preprocessing(df)
        self.treinar_modelo(X_train_tfidf, X_test_tfidf, y_train, y_test)

        return df_aux


    def exec_fluxo_predicao(self, df_aux=None, modelo=False):
        '''
        Aplica pipeline de predição, com ou sem modelo inteligente, partindo de descrições puras:
		1 Fluxo de novos produtos classificados sem modelo, apenas pela pré-classificação, ignora classe 'outros'. (ideal para garantir 0% falsos positivos)
        2 Fluxo de novos produtos classificados com modelo, levando em conta descrições não classificadas na pré-classificação (ideal para uso diário)
        3 Fluxo de predição sobre produtos que não entram no treinamento (apenas para dev, testar df_aux)
        Args:
			df(pandas.DataFrame): Contendo registros a serem classificados pelo modelo treinado (predição)
            modelo(bool)        : Se deve usar o modelo treinado para classificação ou não.
		Retorna:
			df_final(pandas.DataFrame): Produtos classificados na coluna 'classe'.
        '''        
        if df_aux == None:
            encoder_filename = self.config_files['label_encoder_filename']
            le = self.abre(self.path_files, f'{encoder_filename}.pkl')
            try:
                data_classificacao = self.config_files['data_ultima_classificacao']
                data = self.abre(self.path_files, f'{data_classificacao}.pkl')
                df = self.df_produto
                df['dhemi'] = pd.to_datetime(df['dhemi'], format='%Y-%m-%d')
                df = df.loc[df['dhemi'] > str(data).split()[0]]
            except:
                df = self.df_produto
            df = self.filtra_produtos(df)
            df = self.limpa_descricao(df)
            df = self.classifica_descricoes(df)
            df = self.limpa_ucom(df)
            if not modelo:                                
                df_final = df[~df['definicao'].isin(['outros'])]
                df_final['classe'] = le.transform(df_final['definicao'])
                df_final = self.redefine_ucom(df_final)
                return df_final
            else:
                df_aux = df[df['definicao'].isin(['outros'])]
                df = df[~df['definicao'].isin(['outros'])]
                df['classe'] = le.transform(df['definicao'])
                df_aux = self.metodos_pln(df_aux)
                df_aux = self.fazer_predicao_probabilistica(df_aux)
                df_final = pd.concat([df, df_aux])
                df_final = self.redefine_ucom(df_final)
                return df_final
        else:
            return self.fazer_predicao_probabilistica(df_aux)


    def salva_csvs(self, df):
        '''
        Salva arquivos csvs de descrições classificadas e preços por classe em padrão utilizado na visualização chamando métodos de cálculo de preço.
		Se csv e produtos já existir dá um load da variável de ambiente path_csvs para aplicar novos produtos classificados e recalcula preço de classes, se não, cria o primeiro csv.
        Ainda, salva arquivo .pkl com a data mais recente de produtos classificados.
        Args:
			df(pandas.DataFrame): Contendo descrições de produtos classificados para ser cálculado preços.
            cenários
        '''
        try:
            produtos_classificados_filename = self.config_files['produtos_classificados_filename']
            produtos_classificados = pd.read_csv(f'{os.path.join(self.path_files, produtos_classificados_filename)}.csv')    
            df = pd.concat([produtos_classificados, df], ignore_index=True)
        except:
            pass
        
        self.calc_preco(df)
        df = df.round(2)
        df['qcom'] = df['qcom'].astype(int)
        df['vuncom'] = df['vuncom'].astype(str).str.replace('.',",")  
        df['vprod'] = df['vprod'].astype(str).str.replace('.',",")
        df = self.reindex_grupos(df)
        produtos_classificados_filename = self.config_files['produtos_classificados_filename']
        df.to_csv(''.join((self.path_csvs,f'{produtos_classificados_filename}.csv')))
        
        data_classificacao = self.config_files['data_ultima_classificacao']
        data = max(pd.to_datetime(df['dhemi'], format='%Y-%m-%d'))
        self.grava(data, self.path_files, f'{data_classificacao}.pkl')
        

if __name__ == "__main__":
    classificador = ClassificarProdutos()
    #_ = classificador.exec_fluxo_treinamento()
    df_final = classificador.exec_fluxo_predicao(modelo=True)
    classificador.salva_csvs(df_final)
