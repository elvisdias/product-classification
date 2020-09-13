# Author: Elvis Dias <elvisnaopresley@gmail.com>
# Licensed under the MIT License

"""Esse módulo implementa um pipeline de treinamento e teste na categorização de texto para fins de classificar notas fiscais eletrônicas
baseadas apenas por suas descrições de produtos (campo xprod) — visto que não há forma eficiente de classificar através de outros atributos preenchidos.

Esse arquivo é escrito pensando no agendamento de tarefas automáticas, portanto é utilizado em conjunto de outros arquivos: Para sua execução mínima,
é necessário variáveis globais com caminhos de diretório local e nomes de arquivos padrões a serem salvos e consumidos — foi usado um arquivo de configuração .ini
Ainda, requer um arquivo csv contendo alguns valores mínimos contidos numa NFe padrão:
	- Descrição: 'xprod'
	- Valor Unitário: 'vuncom'
	- Data de Emissão: 'dhEmi'
	- NCM: 'ncm'
Outros valores da nota fiscal são utilizados somente para padronizar arquivos de output. 

As variáveis de configuração usadas são:
	- path_csvs                           : caminho para salvar arquivos csvs com as descrições classificadas e preços calculados para cada classe de descrição encontrada
	- path_files_classificacao            : caminho para salvar arquivos de treinamento relativos a clusterização. Estes arquivos são consumidos pela fase de treinamento,
	- produtos_banco_filename             : nome do arquivo csv de input contendo dados da NFe
	- precos_classes                      : nome do arquivo csv de output para criar tabela de preços para classes para uma classe de produtos com dados padrão consumidos por um painel. Formato: {'combustiveis': 'precos_classes_combustiveis', ...}
	- produtos_classificados              : nome do arquivo csv de output contendo todos os produtos que foram classificados em alguma classe válida, este arquivo é lido pela fase de treinamento para serem adicionados novos valores classificados e tem dados padrões consumido por um painel. Formato: {'combustiveis': 'produtos_classificados_combustiveis', ...}
	- produtos_classificados_treinamento  : nome de arquivo pkl de output do treinamento contendo todos os dados com seu histórico de classificação do pipeline. Esse arquivo é usado na fase de treinamento. Formato: {'combustiveis': 'produtos_treinamento', ...}
	- doc2vec                             : nome de arquivo pkl de output do treinamento contendo os vetores do modelo word2vec, utilizados na etapa de classificação para compara descrições por classe. Formato: {'combustiveis': 'doc2vec_combustiveis', ...}
	- word2vec                            : nome de arquivo pkl de output do treinamento contendo o modelo treinado de word2vec, utilizado para classificar as novas amostras sobre os mesmos pesos. Formato: {'combustiveis': 'model_word2vec_combustiveis', ...}
============================================================
Usabilidade:
	O algoritmo é facilmente expansível para mais classes de produtos fora combústiveis, no entanto, o treinamento de classes demanda muito trabalho manual alterando os parâmetros, personalizando o processo de PLN, etc.
	Todas as variáveis alteráveis são setadas nos construtores das classes filhas da HDBSCANClustering. Muitos deles não precisando serem alterados de classe para classe.
	Apesar da customização de parâmetros e processos de limpeza, a execução de treinamento e teste segue um pipeline padrão para todas as classes, executado nas classes fluxo_treinamento e fluxo_predicao:
	
	O pipeline cria durante o processo de classificação e limpeza da classificação várias colunas para cada vez que uma descrição muda de grupo, facilitando a percepção de como o algoritmo altera os resultados (grupo2, grupo3 ...), grupo10 é a coluna de grupos finais.
	ps: não são 10 passos, esse padrão foi mantido do autor original, para manter a relação com o algoritmo original.

	- Treinamento
		:lê csv de produtos e filtra pela classe, limpeza padrão de descrições, limpeza customizada da classe, word embedding, redução de dimensão, clusterização, reduz numero de grupos, tenta encaixar descrições que ficaram de fora no inicio, grava resultados e calula precos para as classes;

	- Teste
		:lê csv de produtos e filtra pela classe, filtra pela data para usar apenas produtos que não já foram utilizados, limpeza padrão de descrições, limpeza customizada da classe, word embedding, encaixa descricoes em classes determinadas, salva novos produtos nas devidas classes e recalula precos para as classes;
Warning:
	Esse arquivo tem um propósito de agendamento somente para a etapa de predição, pois o treinamento é não determinístico por natureza, necessitanto de diversas análises e testes com hiper parâmetros.
==============================================================
Este pipeline foi originalmente compartilhado por https://github.com/alexgand/banco-de-precos
tendo sendo adaptado para o uso do Tribunal de Contas do RN (TCE-RN)
"""

import re
import nltk
import pandas as pd
import numpy as np
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string
import yaml
from string import punctuation
import collections
import unidecode
import hdbscan
import umap
import pickle
import logging
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#from ciex.painelpreco.config_nfe import config_nfe # *PRODUÇÃO EM AIRFLOW*
from config_nfe import * # *DEV*

class HDBSCANClustering():
	'''
	Classe contendo todos os métodos não específicos de classe e necessários para classificar as descrições dos produtos, exibir e salvar resultados. 

	Args:
		...

	'''
	
	def __init__(self, qtd_dimensoes, qtd_dimensoes_umap, qtd_palavras, percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo,\
		quantile_a_retirar_quantidade_palavras_diferentes_no_grupo, quantile_a_retirar_numeros_diferentes_no_grupo, tamanho_minimo_pra_formar_grupo, \
		min_samples, quantile_a_retirar_outliers_dbscan, similarity_minima_pra_juntar_grupos, similarity_minima_pra_encaixar_itens_excluidos_no_final, class_prod):
		
		self.qtd_dimensoes = qtd_dimensoes
		self.qtd_dimensoes_umap = qtd_dimensoes_umap
		self.qtd_palavras = qtd_palavras		
		self.percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo = percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo
		self.quantile_a_retirar_quantidade_palavras_diferentes_no_grupo = quantile_a_retirar_quantidade_palavras_diferentes_no_grupo
		self.quantile_a_retirar_numeros_diferentes_no_grupo = quantile_a_retirar_numeros_diferentes_no_grupo
		self.tamanho_minimo_pra_formar_grupo = tamanho_minimo_pra_formar_grupo
		self.min_samples = min_samples
		self.quantile_a_retirar_outliers_dbscan = quantile_a_retirar_outliers_dbscan
		self.similarity_minima_pra_juntar_grupos = similarity_minima_pra_juntar_grupos
		self.similarity_minima_pra_encaixar_itens_excluidos_no_final = similarity_minima_pra_encaixar_itens_excluidos_no_final
		self.scaler = StandardScaler()
		self.class_prod = class_prod
		self.config_db = config_nfe['FILES']
		self.path_csvs = self.config_db['path_csvs']
		self.path_files = self.config_db['path_files_classificacao']
		self.unidades = ['x','mm','m','cm','ml','g','mg','kg','unidade','unidades','polegada','polegadas',\
		'grama','gramas','gb','mb','l','lr','lt', 'ltr', 'litro', 'ltrs', 'litros','mts','un',\
		'mgml','w','hz','v','gr','lts','lonas','cores','mcg']

	@staticmethod
	def grava(obj, path, filename):
		'''
		Método que salva no path fornecido sarquivos no formato .pkl 
		'''
		
		pkl_file = open(path + filename, 'wb')
		pickle.dump(obj, pkl_file)
		pkl_file.close()
	
	@staticmethod
	def abre(path, filename):
		'''
		Metodo que abre arquivo do path fornecido no formato .pkl
		'''

		pkl_file = open(path + filename, 'rb')
		obj = pickle.load(pkl_file)
		pkl_file.close()
		return obj

	@staticmethod
	def limpa_descricoes(df):
		'''
		Realiza métodos de limpeza de descrição padrão para qualquer classe de produtos.
		Args:
			df(pandas.DataFrame): Contendo pelo menos uma coluna 'xprod' do tipo string
		Retorna:
			df(pandas.DataFrame): Retorna o mesmo objeto recebido, com alterações de texto na coluna 'xprod'
		'''

		nltk.download('stopwords')
		pt_stopwords = set(nltk.corpus.stopwords.words("portuguese"))
		stopwords = set(list(punctuation))
		pt_stopwords.update(stopwords)
		
		#limpa sentencas retirando stopwords, pontuacao e deixa minusculo.
		df['xprod'] = [ ' '.join([word.lower() for word in descr.split() if word.lower() not in pt_stopwords]) for descr in df['xprod'].astype(str)]
		#insere espaco apos / e -, pra no final nao ficar palavras assim: csolucao, ptexto (originais eram c/solucao, p-texto)
		df['xprod'] = df['xprod'].apply(lambda descr: re.sub(r'/|-',r' ',descr))
		#retira pontuacao, /, etc:
		df['xprod'] = df['xprod'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
		#insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
		df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'(\d{1})(\D)',r'\1 \2',x))
		#insere espaco apos letra e numero ex.:c100 pc50
		df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'(\D{1})(\d)',r'\1 \2',x))
		#apaga caracteres de tamanho 1
		df['xprod'] = df['xprod'].apply(lambda x: re.sub(r'\b[a-zA-Z]\b',r'',x))
		#retira espacos duplicados
		df['xprod'] = df['xprod'].apply(lambda x: re.sub(r' +',r' ', x))
		#retira espaco no inicio da frase
		df['xprod'] = df['xprod'].apply(lambda x: x.strip())
		#retira acentos:
		df['xprod'] = df['xprod'].apply(lambda x: unidecode.unidecode(x))
		# remove zeros a esquerda de numeros (02 litros, 05, etc.)
		df['xprod'] = df['xprod'].apply(lambda x: ' '.join([word.lstrip('0') for word in x.split()] ) )
		# remove 'x', pra não diferenciar pneu 275 80 de 275 x 80:
		df['xprod'] = df['xprod'].apply(lambda x: ' '.join([word for word in x.split() if word is not 'x']))
		#se primeira palavra for numero, joga pro final (caso de numeros de referencia que colocam no inicio)
		df['xprod'] = df['xprod'].apply(lambda x: ' '.join(x.split()[1:] + [x.split()[0]]) if ((len(x) > 1) and (x.split()[0].isdigit()) ) else x)
	
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

		preco = df['Valor Unitario de Comercializacao'].dropna()
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
	def reindex_grupos(grupos):
		'''
		Faz um mapeamento dos numeros de grupos determinados pelo algoritmo com um numero sequencial, para fins de melhor usabilidade futura.
		Args:
			grupos(lista): lista contendo os números das classes
		Retorna:
			reindex(dict): Mapeando grupo x to novo numero y
		'''
		reindex = {}
		for i, grupo in enumerate(grupos):
			reindex[grupo] = i+1
		return reindex

	@staticmethod
	def descricoes_grupos(df, grupos, grupox):
		'''
		Método extra para verificar descrições únicas para uma determinada classe, usado para análise de homogeneidade

		Args:
			df(pd.DataFrame): dataframe contendo descrições classificadas
			grupos(lista): Lista contendo os numeros das classes de descrições
			grupox(str): Nome da coluna contendo os numero das classes que se deseja analisar. Preferível que seja a última coluna de classificação.
		
		Retorna:
			df_descrs_unicas(pd.DataFrame): Um dataframe contendo duas colunas, uma com descrições únicas e a outra dizendo a classe de cada descrição
		'''

		groups = []
		descrs = []
		for i, grupo in enumerate(grupos):
			df_grupo = df[df[grupox] == grupo]
			descr_unicas = df_grupo['xprod'].unique()
			for descr in descr_unicas:
				groups.append(i+1)
				descrs.append(descr)
		df_descrs_unicas = pd.DataFrame(zip(groups, descrs), columns=["Grupo", "Descricao"])
		return df_descrs_unicas


	def filtra_df_data(self, df):
		'''
		Filtra produtos a serem classificados no processo de predição para evitar redundância com produtos já classificados.
		A filtragem é feita a partir da data dos produtos já classificados, que é obtida a partir da leitura do arquivo csv dos produtos classificados (caso exista)
		
		Args:
			df(pd.DataFrame): produtos a serem classificados
		
		Retorna:
			df(pd.DataFrame): produtos a serem classificados filtrados ou o mesmo df de input
		'''

		filename = yaml.load(self.config_db['produtos_classificados'], Loader=yaml.FullLoader)
		try:
			df2 = pd.read_csv(''.join((self.config_db['path_csvs'], f'{filename[self.class_prod]}.csv')))
			df2['Data'] = pd.to_datetime(df2['Data'])
			df['dhEmi'] = pd.to_datetime(df['dhEmi'])
			max_date = df['dhEmi'].max()            
			df = df[df['dhEmi'] > max_date]
		except:
			pass

		return df


	def word_embedding(self, df, method='Train'):
		'''
		Implementação alterada do algoritmo word2vec, atribuindo pesos diferentes as palavras de uma descrição. 
		Favorece as mais próximas do início e números.

		Args:
			df(pd.DataFrame): dataframe contendo descrições que serivram de bag of words para modelo word2vec
			method(str): se não informado, por padrão 'Train', treina um modelo de word2vec. Caso não seja 'Train', carrega um modelo de word2vec para fase de teste.
		
		Retorna:
			doc_vectors_std_df: vetores word2vec normalizados
			doc_vectors2: vetores word2vec finais para as descricoes do df fornecido
			doc_vectors3: versão do doc_vectors2 antes da alteração de index, para uso em outro metodo
			model: metodo word2vec treinado para ser salvo por outro metodo
		'''

		sentences = [descr.split()[:self.qtd_palavras] for descr in df['xprod']]
		if method != 'Train':
			w2v = yaml.load(self.config_db['word2vec'], Loader=yaml.FullLoader)
			model = self.abre(self.config_db['path_files_classificacao'], f'{w2v[self.class_prod]}.pkl')
		else:
			model = Word2Vec(sentences, size=self.qtd_dimensoes, min_count=1, workers=-1)

		doc_vectors2 = {}
		for number, sent in enumerate(sentences):
			if len(sent) == 0:
				doc_vectors2[number] = np.zeros(self.qtd_dimensoes,)
			elif len(sent) == 1:
				doc_vectors2[number] = model.wv[sent[0]]
			elif len(sent) > 1:
				pesos = np.array(range(len(sent))[::]) + 1
				pesos = 1 / pesos
				media = []
				divisao = 0
				counter = 0
				for word in sent:            
					if word.isdigit():
						media.append(model.wv[word] * ((pesos[0]+pesos[-1])*(1/2)) )
						divisao += ((pesos[0]+pesos[-1])*(1/2))
					else:
						media.append(model.wv[word] * pesos[counter])
						divisao += pesos[counter]
					counter += 1
				doc_vectors2[number] = np.array(media).sum(axis=0) / divisao
		
		doc_vectors2 = pd.DataFrame(doc_vectors2).T
		doc_vectors3 = doc_vectors2
		doc_vectors2 = doc_vectors2.set_index(df.index) 
		doc_vectors_std_df = pd.DataFrame(self.scaler.fit_transform(doc_vectors2),index=doc_vectors2.index,columns=doc_vectors2.columns)

		return doc_vectors_std_df, doc_vectors2, doc_vectors3, model


	def reducao_dimensao(self, df):
		'''
		Aplica redução de dimensão no df de word2vec utilizando o algoritmo UMAP para facilitar na classificação no processo de cluterização;

		Args:
			df(pd.DataFrame): dataframe contendo os vetores para cada descrição, ja normalizados
		
		Retorna:
			doc_vectors_umap(pd.DataFrame): dataframe com dimensões reduzidas pronto para ser treinado
		'''

		umap_redux = umap.UMAP(n_components=self.qtd_dimensoes_umap, random_state=999, metric='cosine')
		umap_redux.fit(df)
		doc_vectors_umap = umap_redux.transform(X=df)

		return doc_vectors_umap


	def hdbscan_clustering(self, df, doc_vectors_std_df):
		'''
		Treinamento do algoritmo hdbscan aplicando vetores do modelo word2vec para cada descrição.
		Classifica também descrições como outliers

		Args:
			df(pd.DataFrame): dataframe de produtos para serem atribuido os grupos
			doc_vectors_std_df(pd.DataFrame): dataframe com dimensão reduzida para ser treinado
		
		Retorna:
			df(pd.DataFrame): dataframe com produtos atribuidos a um grupo na coluna grupo2
			grupos(lista): lista de indices dos grupos existentes (facilita percepção da limpeza de clusters)
		'''

		clustering = hdbscan.HDBSCAN(min_cluster_size=self.tamanho_minimo_pra_formar_grupo,min_samples=self.min_samples,\
			prediction_data=True, core_dist_n_jobs=-1).fit(doc_vectors_std_df) 
		
		df['grupo'] = clustering.labels_
		
		threshold = pd.Series(clustering.outlier_scores_).quantile(self.quantile_a_retirar_outliers_dbscan)
		outliers = np.where(clustering.outlier_scores_ > threshold)[0]
		df.iloc[outliers,df.columns.get_loc('grupo')] = -2

		grupos = np.unique(clustering.labels_)
		grupos = [grupo for grupo in grupos if grupo >= 0]

		#remove outliers
		exemplars = []
		for exemplar in clustering.exemplars_:
			exemplars.append(np.mean(exemplar,axis=0))
		exemplars_df = pd.DataFrame(exemplars,index=range(len(grupos)))

		map_grupos_exemplars = {}
		df_temp = pd.DataFrame(columns=['sims'])

		for grupo in grupos[:]:
			df2 = df[df['grupo'] == grupo]
			indexes = df2.index
			grupo_vectors = pd.DataFrame(doc_vectors_std_df,index=df.index).loc[indexes]
			
			grupo_do_exemplar = pd.Series(cosine_similarity(grupo_vectors.mean(axis=0).values.reshape(1,-1),exemplars_df)[0])\
				.sort_values(ascending=False).index[0]
			map_grupos_exemplars[grupo] = grupo_do_exemplar

			sims = cosine_similarity(grupo_vectors,exemplars[grupo_do_exemplar].reshape(1,-1))

			df2['sims'] = sims
			df_temp = df_temp.append(df2[['sims']])

		df['sims'] = df_temp
		df['sims'] = df['sims'].replace(np.nan,-1)
		df['grupo2'] = np.where(df['sims'] < 0, -1, df['grupo'])

		grupos = df['grupo2'].unique()
		grupos = [grupo for grupo in grupos if grupo >= 0]

		return df, grupos
	

	def remove_grupos_heterogeneos_primeira_palavra(self, df):
		'''
		Se a 1a palavra nao for a mesma em X% do grupo, exclui o grupo, eh muito heterogeneo
		
		Args:
			df(pd.DataFrame): dataframe de produtos com classes na coluna grupo2 
			
		Retorna: 
			df(pd.DataFrame): dataframe com nova classificação de produtos na coluna grupo4
			grupos(lista): lista contendo os indices de todos os grupos resultantes da etapa.
		'''

		df['grupo3'] = df['grupo2']
		grupos = sorted(df['grupo3'].unique())
		grupos = [grupo for grupo in grupos if grupo >=0]

		grupos_homogeneos = []
		for grupo in grupos:
			df2 = df[df['grupo3'] == grupo]
			if len(df2) > 0:
				if ( df2['xprod'].apply(lambda x: x.split()[0] if (len(x.split()) > 0) else np.random.random()).value_counts().iloc[0] / len(df2) )\
					 > self.percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo:
					grupos_homogeneos.append(grupo)

		df['grupo4'] = df['grupo3'].isin(grupos_homogeneos)
		df['grupo4'] = np.where(df['grupo4'], df['grupo3'], -1)

		grupos = sorted(df['grupo4'].unique())
		grupos = [grupo for grupo in grupos if grupo >=0] 

		return df, grupos


	def remove_grupos_heterogeneos_contagem_palavras(self, df):
		'''
		Exclui grupos homogeneos pela contagem de palavras diferentes;
		
		Args:
			df(pd.DataFrame): dataframe de produtos com classes na coluna grupo4 
			
		Retorna: 
			df(pd.DataFrame): dataframe com nova classificação de produtos na coluna grupo6
			grupos(lista): lista contendo os indices de todos os grupos resultantes da etapa.
		'''

		df['qtd_palavras_diferentes'] = df['xprod'].apply(lambda x: len(set( [item for sublist in [sent.split()[:self.qtd_palavras] for sent in x] for item in sublist] )))
		qtd_palavras_por_grupo = df.groupby('grupo4')['qtd_palavras_diferentes'].median()
		qtd_palavras_por_grupo = qtd_palavras_por_grupo.sort_values()
		qtd_max_palavras_diferentes_no_grupo = int(qtd_palavras_por_grupo.quantile(self.quantile_a_retirar_quantidade_palavras_diferentes_no_grupo))

		df['qtd_median_palavras_dif_grupo'] = df['grupo4'].map(qtd_palavras_por_grupo)
		df['grupo5'] = np.where(df['qtd_median_palavras_dif_grupo'] > qtd_max_palavras_diferentes_no_grupo, -1, df['grupo4'])

		grupos = df['grupo5'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0]

		#pulei passo do grupo6
		df['grupo6'] = df['grupo5']

		return df, grupos


	def remove_grupos_heterogeneos_contagem_numeros(self, df):
		'''
		Exclui grupos que tem muitos números diferentes nas sentences;
		
		Args:
			df(pd.DataFrame): dataframe de produtos com classes na coluna grupo4 e grupo6
			
		Retorna: 
			df(pd.DataFrame): dataframe com nova classificação de produtos na coluna grupo7
			grupos(lista): lista contendo os indices de todos os grupos resultantes da etapa.
		'''

		df['qtd_numeros_diferentes'] = df['xprod'].apply(lambda x: len(set( [item for sublist in [sent.split()[:self.qtd_palavras] for sent in x] for item in sublist if item.isdigit()   ] ) ))
		df['qtd_numeros_diferentes'] = np.where(df['qtd_numeros_diferentes'] == 0, 0, df['qtd_numeros_diferentes']-1)
		qtd_numeros_por_grupo = df.groupby('grupo4')['qtd_numeros_diferentes'].median()
		qtd_numeros_por_grupo = qtd_numeros_por_grupo[qtd_numeros_por_grupo > 0]
		qtd_numeros_por_grupo = qtd_numeros_por_grupo.sort_values()
		qtd_max_numeros_diferentes_no_grupo = int(qtd_numeros_por_grupo.quantile(self.quantile_a_retirar_numeros_diferentes_no_grupo))
		df['qtd_median_numeros_dif_grupo'] = df['grupo4'].map(qtd_numeros_por_grupo)

		df['grupo7'] = np.where(df['qtd_median_numeros_dif_grupo'] > qtd_max_numeros_diferentes_no_grupo, -1, df['grupo6'])

		grupos = df['grupo7'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0] 

		return df, grupos


	def junta_grupos_semelhantes(self, df, doc_vectors):
		'''
		Junta grupos semelhantes baseado na similaridade do coseno e no parametro 'similarity_minima_pra_juntar_grupos'
		
		Args: 
			df(pd.DataFrame): dataframe de produtos com classes na coluna grupo7
			doc_vectors(pd.DataFrame): doc_vectors de treinamento antes do reindex: 'doc_vectors3' retornado pelo metodo word_embedding
		
		Retorna: 
			df(pd.DataFrame): dataframe com nova classificação de produtos na coluna grupo9
			grupos(lista): lista contendo os indices de todos os grupos resultantes da etapa.
		'''

		doc_vectors = doc_vectors.set_index(df.index) 
		doc_vectors_grupos = {}
		
		grupos = df['grupo7'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0] 
		
		for grupo in grupos:
			indices = df[df['grupo7'] == grupo].index
			doc_vectors_grupos[grupo] = doc_vectors.loc[indices]
			doc_vectors_grupos[grupo] = doc_vectors_grupos[grupo].mean(axis=0)   

		doc_vectors_grupos = pd.DataFrame(doc_vectors_grupos).T
		doc_vectors_grupos_std = pd.DataFrame(self.scaler.transform(doc_vectors_grupos),index=doc_vectors_grupos.index,columns=doc_vectors_grupos.columns)

		grupos_similarities = cosine_similarity(doc_vectors_grupos_std)
		grupos_similarities = pd.DataFrame(grupos_similarities,index=doc_vectors_grupos.index,columns=doc_vectors_grupos.index)

		#junta os grupos:
		grupos_similares = []
		for grupo in grupos_similarities:
			agrupar_df = grupos_similarities[grupo].sort_values(ascending=False)
			agrupar_df = agrupar_df[agrupar_df >= self.similarity_minima_pra_juntar_grupos]
			grupos_similares.append(list(agrupar_df.index))

		novo_grupo = 0
		mapeamento_grupos = {}
		for mini_grupo in grupos_similares:
			if len(mini_grupo) == 1:
				mapeamento_grupos[mini_grupo[0]] = novo_grupo
			else:
				for grupo in mini_grupo:
					if grupo not in mapeamento_grupos.keys():
						for mini_grupo2 in grupos_similares:
							if grupo in mini_grupo2:
								mapeamento_grupos[grupo] = novo_grupo
								for grupo2 in mini_grupo2:
									if grupo2 not in mapeamento_grupos.keys():
										mapeamento_grupos[grupo2] = novo_grupo
			novo_grupo += 1

		df['grupo9'] = df['grupo7'].map(mapeamento_grupos)
		df['grupo9'] = df['grupo9'].fillna(-1)
		df['grupo9'] = df['grupo9'].astype(int)

		grupos = df['grupo9'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0] 

		return df, grupos


	def encaixa_outsiders(self, df, doc_vectors, method='Train'):
		'''
		Repassar as sentences de classe -1 e tentar encaixá-las nos grupos formados baseando na similaridade do coseno e no parametro 'similarity_minima_pra_encaixar_itens_excluidos_no_final'

		Args:
			df(pd.DataFrame): dataframe contendo classes do grupo9 se for treinamento ou só dataframe qualquer para teste
			doc_vectors(pd.DataFrame): doc_vectors das descrições
			method(str): padrão 'Train'. Indica qual metodologia deve adotar para separar indexes inclusos ou exlusos. 

		Retorna:
			df(pd.DataFrame): dataframe com nova classificação de produtos na coluna grupo10 (reindexado para numeros sequenciais)
			grupos(lista): lista contendo os indices (reindexados) de todos os grupos resultantes da etapa.
		'''		 
		
		doc_vectors_grupos_finais={}
		if method != 'Train':
			df['grupo9'] = -1

			df_train_file = yaml.load(self.config_db['produtos_classificados_treinamento'],Loader=yaml.FullLoader)
			d2v = yaml.load(self.config_db['doc2vec'], Loader=yaml.FullLoader)
			df_train = self.abre(self.config_db['path_files_classificacao'], f'{df_train_file[self.class_prod]}.pkl')
			doc_vectors2 = self.abre(self.config_db['path_files_classificacao'], f'{d2v[self.class_prod]}.pkl')

			excluidos_index =  doc_vectors.index
			incluidos_index =  df_train[df_train['grupo9'] >= 0].index

			doc_vectors_excluidos = doc_vectors
			doc_vectors_incluidos = doc_vectors2.loc[incluidos_index]

			grupos = df_train['grupo9'].unique()
			grupos = [grupo for grupo in grupos if grupo>=0]

			for grupo in grupos:
				df2 = df_train[df_train['grupo9'] == grupo]
				doc_vectors_grupos_finais[grupo] = doc_vectors_incluidos.loc[df2.index]
				doc_vectors_grupos_finais[grupo] = doc_vectors_grupos_finais[grupo].values.mean(axis=0)

		else:
			excluidos_index =  df[df['grupo9'] == -1].index
			incluidos_index =  df[df['grupo9'] >= 0].index

			doc_vectors_excluidos = doc_vectors.loc[excluidos_index]
			doc_vectors_incluidos = doc_vectors.loc[incluidos_index]
			
			grupos = df['grupo9'].unique()
			grupos = [grupo for grupo in grupos if grupo>=0]
			
			for grupo in grupos:
				df2 = df[df['grupo9'] == grupo]
				doc_vectors_grupos_finais[grupo] = doc_vectors_incluidos.loc[df2.index]
				doc_vectors_grupos_finais[grupo] = doc_vectors_grupos_finais[grupo].values.mean(axis=0)
		
		doc_vectors_grupos_finais = pd.DataFrame(doc_vectors_grupos_finais).T
		compara = cosine_similarity(doc_vectors_excluidos.loc[excluidos_index],doc_vectors_grupos_finais.values)
		compara = pd.DataFrame(compara,index=excluidos_index, columns=grupos)
		similarity_do_grupo_mais_parecido = compara.max(axis=1)
		grupo_mais_parecido = compara.idxmax(axis=1)

		encaixar_excluidos = pd.Series( np.where(similarity_do_grupo_mais_parecido >=\
			self.similarity_minima_pra_encaixar_itens_excluidos_no_final,\
			grupo_mais_parecido, -1), index= similarity_do_grupo_mais_parecido.index)
		
		df['grupo10'] = encaixar_excluidos
		df['grupo10'] = df['grupo10'].fillna(-1)
		df['grupo10'] = np.where(df['grupo10'] == -1, df['grupo9'], df['grupo10'])
		
		grupos = df['grupo10'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0] 

		reindex = self.reindex_grupos(grupos)
		df['grupo10'] = df['grupo10'].map(reindex)
		
		grupos = df['grupo10'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0]

		return df, grupos


	def mostra_resultados_treinamento(self, df):
		'''
		Método auxiliar ao treinamento para ilustrar a % de descrições que foram desconsideradas pelo pipeline de classificação
		e imprime amostras para cada classe final, ilustrando resultados.

		Args:
			df(pd.DataFrame): dataframe contendo todas as colunas 'grupo' antes do reindex de classes
		'''

		grupos = df['grupo10'].unique()
		grupos = [grupo for grupo in grupos if grupo >=0] 

		values = [-1, -2] #verificar os que ficaram de fora (-1) e foram considerados outliers (-2)
		out_inicio = df[df.grupo.isin(values)]
		out_fim = df[df['grupo10']==-1]
		resu = (len(out_inicio)+len(out_fim))/(len(df['ncm'].values.tolist()))*100

		print(f'Ficaram fora no início: {len(out_inicio)}')
		print(f'Ficaram fora no fim: {len(out_fim)}')
		print(f'Total de registros de fora (%): {resu}')
		print('--------------------------------------------\n')
		print(self.print_exemplos_grupos_v2_aleatorio(df=df,grupos=grupos,grupox='grupo10',cols=['xprod', 'vuncom'],qtd_palavras=self.qtd_palavras,unidades=self.unidades))
	

	def calc_preco(self, df):
		'''
		Calcula preco médio, minimo e máximo para a coluna de precos 'Valor Unitario de Comercializacao' para cada classe presente na coluna 'id'
		salva um arquivo csv com os preços para cada classe em formato brasileiro de float (usado em visualização)

		Args:
			df(pd.DataFrame): dataframe contendo produtos com coluna de preços 'Valor Unitario de Comercializacao' e 'id' (determinado no método salva_novos_produtos)
		'''
		
		df_list_grupos = []
		groups = []
		group_names = []
		media = []
		mediana = []
		media_saneada = []
		precos_max = []
		precos_min = []
		coluna_preco = 'Valor Unitario de Comercializacao'
		coluna_grupo = 'id'

		df[coluna_preco] = df[coluna_preco].str.replace(',',".")
		df[coluna_preco] = pd.to_numeric(df[coluna_preco])
		
		grupos = df[coluna_grupo].unique()
		grupos = [grupo for grupo in grupos if grupo >=0]
		
		for i, grupo in enumerate(grupos):
			df_grupo = df[df[coluna_grupo] == grupo]
			df_list_grupos.append(df_grupo)	
			
			#S = E – M sobrepreço estimado - referencial mercado
			media.append(round(df_grupo[coluna_preco].mean(),3))
			mediana.append(round(df_grupo[coluna_preco].median(),3))
			
			nova_media, preco_max, preco_min = self.calc_media_saneada(df_grupo)
			
			media_saneada.append(nova_media)
			precos_max.append(preco_max)
			precos_min.append(preco_min)
		
			descricao = self.get_nome_representacao_do_grupo(df=df[df[coluna_grupo]==grupo])
			descricao = ' '.join(descricao)
			group_names.append(descricao)
			groups.append(i+1)
		
		df_precos = pd.DataFrame(zip(groups, group_names, media, mediana, precos_max, precos_min, media_saneada),\
								columns=["id","Descricao", "Media", "Mediana", "Max", "Min", "Media Saneada"])
		
		df_precos = df_precos.round(2)
		df_precos['Media'] = df_precos['Media'].astype(str).str.replace('.',",")  
		df_precos['Mediana'] = df_precos['Mediana'].astype(str).str.replace('.',",")  
		df_precos['Max'] = df_precos['Max'].astype(str).str.replace('.',",")  
		df_precos['Min'] = df_precos['Min'].astype(str).str.replace('.',",")  
		df_precos['Media Saneada'] = df_precos['Media Saneada'].astype(str).str.replace('.',",")
		
		df_precos.set_index('id', inplace=True)
		filename = yaml.load(self.config_db['precos_classes'], Loader=yaml.FullLoader)
		df_precos.to_csv(''.join((self.path_csvs, f'{filename[self.class_prod]}.csv')))

	
	def get_nome_representacao_do_grupo(self, df):
		'''
		Obtém uma descrição para cada classe de produtos a partir dos padrões de suas descrições 

		Args:
			df(pd.DataFrame): dataframe filtrado por grupo de interesse
		
		Retorna:
			representacao_grupo(lista): palavras descritivas do grupo de interesse
		'''
		
		percentual_pra_manter_palavra_na_representacao = 0.50
		#pega cada palavra e ve as que mais se repetem nas sentences
		#fica com aquelas que estao em mais do que X% das sentences
		sentences = [sent for sent in df['Descricao do Produto']]
		if len(set(sentences)) == 1: #ou seja, todos os itens sao iguais.
			representacao_grupo = [word for word in set(sentences)][0].split()
		else:
			palavras_series = df['Descricao do Produto'].str.split()
			palavras_series = palavras_series.apply(lambda x: x[:self.qtd_palavras])
			contagem_palavras_nas_sentences = {}
			palav = [item for sublist in palavras_series for item in sublist]
			palavras = sorted( set(palav), key=palav.index) #pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
			for palavra in palavras:
				contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()
		
			contagem_palavras_nas_sentences = pd.Series(contagem_palavras_nas_sentences)
			contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df)
			contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
			representacao_grupo = list(contagem_palavras_nas_sentences.index)
		#reordenacao:
		primeiras_palavras = [word for word in representacao_grupo if ((not word.isdigit()) and word not in self.unidades)]
		meio_palavras = [word for word in representacao_grupo if word.isdigit()]
		ultimas_palavras = [word for word in representacao_grupo if word in self.unidades]
		
		# intercala numeros e unidades:
		if len(meio_palavras) == len(ultimas_palavras):
			result = [None]*(len(meio_palavras)+len(ultimas_palavras))
			result[::2] = meio_palavras
			result[1::2] = ultimas_palavras
			meio_palavras = result
			ultimas_palavras = []
		else:
			# if 'x' in representacao_grupo:
			if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
				ultimas_palavras = [word for word in ultimas_palavras if word != 'x'] #retira o 'x', vai inserir abaixo:
				meio_palavras.insert(1,'x') #insere o 'x' apos o 1o numero
		if len(meio_palavras) == 0:
			representacao_grupo = primeiras_palavras #daih nao coloca unidades
		else:
			representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
		
		return representacao_grupo

	
	def print_exemplos_grupos_v2_aleatorio(self, df, grupos, grupox, cols, qtd_palavras,unidades):
		'''
		Imprime descrições dos grupos e as descrições que os compoem. Auxiliar no processo de treinamento
		Args:
			...
		'''
		
		for grupo in grupos:
			df_mostrar = df[df[grupox] == grupo]
			print('\nGrupo:',grupo,'len:',len(df_mostrar))
			print(self.get_nome_representacao_do_grupo(df=df[df[grupox]==grupo]))
			print(df_mostrar[cols])
	

	def grava_resultados_treinamento(self, df, doc_vectors, model):
		'''
		Salva no caminho da variável path_files os arquivos de treinamento que serão usados para predição
		Args:
			df(pd.DataFrame): dataframe completo do processo de treinamento, com o histórico do pipeline de classificação
			doc_vectors(pd.DataFrame): o doc_vectors das descrições de treinaemnto
			model(gensim.models): modelo word2vec treinado
		'''
		
		filename = yaml.load(self.config_db['produtos_classificados_treinamento'], Loader=yaml.FullLoader)
		self.grava(df, self.path_files, f'{filename[self.class_prod]}.pkl')
		filename = yaml.load(self.config_db['doc2vec'], Loader=yaml.FullLoader)
		self.grava(doc_vectors, self.path_files, f'{filename[self.class_prod]}.pkl')
		filename = yaml.load(self.config_db['word2vec'], Loader=yaml.FullLoader)
		self.grava(model, self.path_files, f'{filename[self.class_prod]}.pkl')


	def salva_novos_produtos(self, df, method='Train'):
		'''
		Salva o arquivo csv final de classificação com colunas renomeadas para melhor exibição. Se arquivo for de predição da um append no arquivo que já existia, aumentando as amostras por classe.
		Args:
			df(pd.DataFrame): dataframe completo de treinamento, com todas as colunas lidas do csv original
			method(Str): padrão 'Train'. Determina se vai haver tentativa de leitura de arquivo já existente para adicionar novos produtos às classes.
		'''
		
		cols_interesse = ['idnfe', 'dhemi', 'xnome', 'ncm', 'cean', 'ucom', 'qcom', 'xprod', 'vuncom', 'vprod', 'grupo10']
		
		df.reset_index(inplace=True)
		df = df[df['grupo10'] >= 0]
		df = df[cols_interesse]
		df = df.round(2)
		df['grupo10'] = df['grupo10'].astype(int)
		df['idnfe'] = df['idnfe'].astype(str)
		df['qcom'] = df['qcom'].astype(str).str.replace('.',",")
		df['vuncom'] = df['vuncom'].astype(str).str.replace('.',",")
		df['vprod'] = df['vprod'].astype(str).str.replace('.',",")

		df.rename(columns={'grupo10':'id', 'dhemi':'Data', 'ncm':'NCM', 'xnome':'Orgao', 'xprod':'Descricao do Produto',\
								'cean':'Codigo de Barras', 'qcom':'Quantidade Comercial',\
								'ucom':'Unidade de Comercializacao', 'vuncom':'Valor Unitario de Comercializacao', 'vprod':'Total'}, inplace=True)
		
		filename = yaml.load(self.config_db['produtos_classificados'], Loader=yaml.FullLoader)
		
		if method != 'Train':
			try:
				df = pd.read_csv(''.join((self.path_csvs,f'{filename[self.class_prod]}.csv')))
				df = df.append(df, ignore_index=True)
				df.to_csv(''.join((self.path_csvs,f'{filename[self.class_prod]}.csv')), columns=['idnfe', 'Data', 'NCM', 'Codigo de Barras', 'Orgao', 'Unidade de Comercializacao', 'Quantidade Comercial', 'Descricao do Produto', 'Valor Unitario de Comercializacao', 'Total', 'id'])
			except:			
				df.to_csv(''.join((self.path_csvs,f'{filename[self.class_prod]}.csv')))
		else:
			df.to_csv(''.join((self.path_csvs,f'{filename[self.class_prod]}.csv')))
		
		return df


	def fluxo_treinamento(self, df):
		'''
		Aplica pipeline de treinamento sequencial comum entre classes 
		Args:
			df(pd.DataFrame): dataframe com descrições dos produtos já limpa para a classe correspondente
		'''
		
		doc_vectors_std_df, doc_vectors, doc_vectors2, model  = self.word_embedding(df)
		doc_vectors_std_df_umap = self.reducao_dimensao(doc_vectors_std_df)
		df, _ = self.hdbscan_clustering(df, doc_vectors_std_df_umap)
		df, _ = self.remove_grupos_heterogeneos_primeira_palavra(df)
		df, _ = self.remove_grupos_heterogeneos_contagem_palavras(df)
		df, _ = self.remove_grupos_heterogeneos_contagem_numeros(df)
		self.junta_grupos_semelhantes(df, doc_vectors2)
		df, _ = self.encaixa_outsiders(df, doc_vectors)
		self.grava_resultados_treinamento(df, doc_vectors, model)
		df = self.salva_novos_produtos(df)
		self.calc_preco(df)


	def fluxo_predicao(self, df_teste):
		'''
		Aplica pipeline de predição sequencial comum entre classes 
		conta com a existência dos arquivos de treinamento no caminho de path_files
		
		Args:
			df(pd.DataFrame): dataframe com descrições dos produtos já limpa para a classe correspondente
		'''

		_, doc_vectors, _ , _  = self.word_embedding(df_teste, method='Teste')
		df, _ = self.encaixa_outsiders(df_teste, doc_vectors, method='Teste')
		df_appended = self.salva_novos_produtos(df, method='Teste')
		self.calc_preco(df_appended)


class CombustiveisClustering(HDBSCANClustering):

	def __init__(self, qtd_dimensoes = 200, qtd_dimensoes_umap = 10, qtd_palavras = 10, percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo = 0.70,\
		quantile_a_retirar_quantidade_palavras_diferentes_no_grupo = 0.95, quantile_a_retirar_numeros_diferentes_no_grupo = 0.95, tamanho_minimo_pra_formar_grupo = 35, \
		min_samples = 1, quantile_a_retirar_outliers_dbscan = 0.99, similarity_minima_pra_juntar_grupos = 0.65, similarity_minima_pra_encaixar_itens_excluidos_no_final = 0.85,\
		class_prod='combustiveis'):
		
		super().__init__(qtd_dimensoes, qtd_dimensoes_umap, qtd_palavras, percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo,\
		quantile_a_retirar_quantidade_palavras_diferentes_no_grupo ,quantile_a_retirar_numeros_diferentes_no_grupo, tamanho_minimo_pra_formar_grupo,\
		min_samples, quantile_a_retirar_outliers_dbscan, similarity_minima_pra_juntar_grupos, similarity_minima_pra_encaixar_itens_excluidos_no_final, class_prod)


	def read_csv(self):
		'''
		Lê arquivo csv de produtos contendo a coluna NCM e realiza a filtragem dos produtos para manter somente combustíveis
		
		Retorna:
			df(pd.DataFrame): dataframe de produtos combustíveis a serem classificados
		'''

		filename = self.config_db['produtos_banco_filename']
		df = pd.read_csv(''.join((self.path_csvs,f'{filename}.csv')))
		most_common = [ncm for ncm, qtd in collections.Counter(df['ncm'].values.tolist()).most_common(2)]
		df = df[df['ncm'].apply(lambda ncm: ncm in most_common)]

		return df 


	@staticmethod
	def remove_stopwords_combustiveis(df):
		'''
		Realiza limpeza para a classe de produtos combustíveis. Cada classe exige uma rotina diferente de limpeza adicional à padrão.

		Args:
			df(pd.DataFrame): dataframe com descrições limpas pelo método padrão
			
		Retorna:
			df(pd.DataFrame): dataframe com descrições limpas por completo para combustíveis
		'''

		nltk.download('stopwords')
		pt_stopwords = set(nltk.corpus.stopwords.words("portuguese"))
		pt_stopwords.add('oleo')
		pt_stopwords.add('petrobras')
		pt_stopwords.add('bs')
		pt_stopwords.add('br')
		pt_stopwords.add('tipo')

		df['xprod'] = [ ' '.join([word for word in descr.split() if word not in pt_stopwords]) for descr in df['xprod'].astype(str)]

		return df

	
	def executa_treinamento(self):
		'''
		Mescla processo de treinamento não específico da classe com específico, sendo o único método responsável por executar todo o fluxo de treinamento.
		'''
		df = self.read_csv()
		df = self.limpa_descricoes(df)
		df = self.remove_stopwords_combustiveis(df)
		self.fluxo_treinamento(df)


	def executa_predicao(self):
		'''
		Mescla processo de predição não específico da classe com específico, sendo o único método responsável por executar todo o fluxo de predição.
		'''

		df_teste = self.read_csv()
		df_teste = self.filtra_df_data(df_teste)
		df_teste = self.limpa_descricoes(df_teste)
		df_teste = self.remove_stopwords_combustiveis(df_teste)
		self.fluxo_predicao(df_teste)		


if __name__ == '__main__':
	combust = CombustiveisClustering()
	combust.executa_treinamento()
	combust.executa_predicao()