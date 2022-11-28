import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pulp
from scipy.spatial.distance import minkowski
from itertools import product
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA


class PatientMatcher():

	def __init__(self, excel_filepath, disorder):#, **read_xls_args):
		'''
		Reads in an excel file of patient comorbidity data upon which to perform matching.

		Parameters
		----------
		excel_file: string
		  The file path to the excel file containing patient data

		Returns
		-------
		matcher: PatientMatcher
		'''
		data = pd.read_csv(excel_filepath, encoding = "ISO-8859-1", low_memory=False)
		data = data.set_index('SubjectID')

		data = data[[col for col in data.columns.tolist() if (col[-3:]=='CUR' and col[-5:]!='RECUR') or (col[-2:]=='DX')]]

		data = data.loc[:, data.std() > 0]
		data = data[data[disorder].notna()]
		data = data[data.notna().all(axis=1)]

		self.var_cols = data.columns.drop(disorder)
		self.bin_cols = data.columns[data.isin([0, 1]).all()].tolist()

		self.mean, self.std = data[self.var_cols].mean(
		), data[self.var_cols].std()

		self.data = data
		self.disorder_str = disorder

	def get_best_matches(self, feature_weights: dict = None, norm_order: int = 2) -> pd.DataFrame:
		'''
		Calculates the best match for each disorder patient
		given no disorder patient can be matched to the same control patient.

		Parameters
		----------
		feature_weights: dict (optional)
		  A dict of {feature (str): weight (float)} to control importance of each input variable
		  to the matching algorithm. Weights default to 1.0 if not specified.

		norm_order: int (default: 2)
		  Controls the norm order of the Minkowski distance metric.
		  A norm order of 2 is equivalent to Euclidean distance.

		Returns
		-------
		matches: Pandas Dataframe
		  Contains the optimal matches found by the matching algorithm,
		  along with the matches' dissimilarity scores.
		'''

		# split into disorder and psychiatric control
		self.disorder = self.data.loc[self.data[self.disorder_str].values==1, self.var_cols].sub(
			self.mean).div(self.std)
		self.control = self.data.loc[self.data[self.disorder_str].values==0, self.var_cols].sub(
			self.mean).div(self.std)

		fw = pd.Series(1., index=self.var_cols)
		if feature_weights:
			for feature, weight in feature_weights.items():
				fw.loc[feature] = weight

		d = pd.Series({
			(control_id, disorder_id): minkowski(self.control.loc[control_id], self.disorder.loc[disorder_id], p=norm_order, w=fw)
			for control_id, disorder_id in product(self.control.index, self.disorder.index)
		})
		self.disorder_str = self.disorder_str.split('CUR')[0].lower()+'_id'      
		d.index.names = ['control_id', self.disorder_str]
		self.dissimilarity_matrix = d.unstack(d.index.names[1])

		prob = pulp.LpProblem('match', pulp.LpMinimize)

		x = np.array([
			[pulp.LpVariable(f'({i},{j})', cat='Binary')
			 for j in range(self.dissimilarity_matrix.shape[1])]
			for i in range(self.dissimilarity_matrix.shape[0])
		])

		# a control patient can be matched to no more than one with disorder
		for i in range(x.shape[0]):
			prob += pulp.lpSum(x[i, :]) <= 1

		# each disorder patient must have exactly one match
		for j in range(x.shape[1]):
			prob += pulp.lpSum(x[:, j]) == 1

		# add the objective function to minimize
		prob += pulp.lpSum(np.multiply(self.dissimilarity_matrix.values, x).flatten())

		status = prob.solve()
		print('Matching status:', pulp.LpStatus[status])

		matches = pd.DataFrame(
			x,
			index=self.dissimilarity_matrix.index,
			columns=self.dissimilarity_matrix.columns
		).applymap(pulp.value).idxmax().rename('control_id').to_frame()
		matches['score'] = [
			self.dissimilarity_matrix.loc[cont, bp]
			for bp, cont in matches.control_id.iteritems()
		]
		matches['exact_feature_matches'] = [
			sum(self.data.loc[bp, self.bin_cols] == self.data.loc[cont, self.bin_cols])
			for bp, cont in matches.control_id.iteritems()
		]

		return matches

	def decomp_viz(self, matches: pd.DataFrame, decomp_method: str = 'PCA', random_state: int = 42, **algo_kwargs):
		'''
		A "sanity checker" to visualize how well the matching algorithm performed. May only be called after calling get_best_matches.

		Parameters
		----------
		matches: Pandas Dataframe
		  The output of get_best_matches.

		decomp_method: str (default: 'PCA')
		  Controls which Scikit-Learn algorithm is used to visualize the patient data in 2 dimensions.
		  Must be 'PCA' (Principal Component Analysis, 'MDS' (Multi-Dimensional Scaling),
		  or 'TSNE' (T-Distributed Stochastic Neighbor Embedding).

		random_state: int or None (default: 42)
		  If not None, sets the random seed of the decomposition algorithm for reproducible results.

		**algo_kwargs:
		  Other keyword arguments to be passed to the decomposition algorithm.

		Returns
		-------
		algo: Scikit-Learn object
		  The decomposition algorithm fitted to the data.
		'''

		if decomp_method == 'PCA':
			algo = PCA
		elif decomp_method == 'MDS':
			algo = MDS
		elif decomp_method == 'TSNE':
			algo = TSNE
		else:
			raise ValueError(
				f'Unrecognized decomposition method "{decomp_method}"')

		algo = algo(
			n_components=2,
			random_state=random_state,
			**algo_kwargs
		)

		decomp = pd.DataFrame(
			algo.fit_transform(self.data[self.var_cols].sub(
				self.mean).div(self.std)),
			index=self.data.index,
			columns=['dim1', 'dim2']
		)

		# plot points
		ax = plt.subplots(1, 2, figsize=(20, 10))[1]
		ax[0].set_title(
			f'Visualization of matches with {str(algo).split("(")[0]}')
		ax[1].set_title(
			'Distribution of match dissimilarity (lower is better)')
		decomp[decomp.index.isin(self.control.index)].plot.scatter(
			'dim1', 'dim2', c='C0', ax=ax[0], label='Control')
		decomp[decomp.index.isin(self.disorder.index)].plot.scatter(
			'dim1', 'dim2', c='C1', ax=ax[0], label=self.disorder_str.split('_')[0].capitalize())

		matches.score.hist(ax=ax[1], bins=50)
		ax[1].set_xlabel('Dissimilarity score')
		ax[1].set_ylabel('Frequency')

		# connect the dots
		for match in matches.control_id.iteritems():
			ax[0].add_line(
				plt.Line2D(
					xdata=decomp.loc[np.array(match).T, 'dim1'],
					ydata=decomp.loc[np.array(match).T, 'dim2'],
					color='grey',
					alpha=.3
				)
			)
		return algo