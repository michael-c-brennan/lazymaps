
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.close('all')


class blank_class:
    pass

def latex_float(f):
    float_str = "{0:.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

# Taking averages of data
def ave_data(ss, model, num_trials, string):
    holder = blank_class()

    ave_losses = 0
    min_losses = []

    ave_traces = 0
    min_traces = []

    ave_is_traces = 0
    min_is_traces = []

    ave_neg_elbos = 0
    min_neg_elbos = []
    diagnostics = []
    for trial in range(num_trials):


        file_name = string+'_'+ model+\
                    '_sample_size_'+str(ss)+\
                    '_trial_'+str(trial)+\
                    '_diagnostics'

        diagnostics.append(np.loadtxt(file_name))

        file_name = string+'_'+ model+\
                    '_sample_size_'+str(ss)+\
                    '_trial_'+str(trial)+\
                    '_training_data'

        data = np.loadtxt(file_name)

        ave_losses += data[0]
        min_losses.append(np.min(data[0]))

        ave_traces += data[1]
        min_traces.append(np.min(data[1]))

        ave_is_traces += data[2]
        min_is_traces.append(np.min(data[2]))

        ave_neg_elbos += data[3]
        min_neg_elbos.append(np.min(data[3]))

    steps = data[4]

    holder.diagnostics = diagnostics

    holder.steps = steps

    holder.ave_losses = ave_losses/num_trials

    holder.ave_traces = ave_traces/num_trials

    holder.ave_is_traces = ave_is_traces/num_trials

    holder.ave_neg_elbos = ave_neg_elbos/num_trials

    holder.mins = [min_losses, min_traces, min_is_traces, min_neg_elbos]

    return holder


def lazy_ave_data(lr, nll, ss, model, num_trials, string):

    holder = blank_class()

    ave_losses = 0
    min_losses = []

    ave_traces = 0
    min_traces = []

    ave_is_traces = 0
    min_is_traces = []

    ave_neg_elbos = 0
    min_neg_elbos = []

    eigvals = []
    diagnostics = []
    for trial in range(num_trials):

        file_name = string+'_'+ model+'_rank_'+str(lr) +\
                    '_num_lazy_layers_'+str(nll) +\
                    '_sample_size_'+str(ss) +\
                    '_trial_'+str(trial) +\
                    '_eigvals'

        eigvals.append(np.loadtxt(file_name))

        file_name = string+'_'+ model+'_rank_'+str(lr) +\
                    '_num_lazy_layers_'+str(nll) +\
                    '_sample_size_'+str(ss) +\
                    '_trial_'+str(trial) +\
                    '_diagnostics'

        diagnostics.append(np.loadtxt(file_name))

        file_name = string+'_'+ model+'_rank_'+str(lr) +\
                    '_num_lazy_layers_'+str(nll) +\
                    '_sample_size_'+str(ss) +\
                    '_trial_'+str(trial) +\
                    '_training_data'

        data = np.loadtxt(file_name)

        ave_losses += data[0]
        min_losses.append(np.min(data[0]))

        ave_traces += data[1]
        min_traces.append(np.min(data[1]))

        ave_is_traces += data[2]
        min_is_traces.append(np.min(data[2]))

        ave_neg_elbos += data[3]
        min_neg_elbos.append(np.min(data[3]))

    holder.diagnostics = diagnostics
    holder.eigvals = eigvals

    steps = data[4]

    holder.steps = steps

    holder.ave_losses = ave_losses / num_trials

    holder.ave_traces = ave_traces / num_trials

    holder.ave_is_traces = ave_is_traces / num_trials

    holder.ave_neg_elbos = ave_neg_elbos / num_trials

    holder.mins = [min_losses, min_traces, min_is_traces, min_neg_elbos]

    return holder


num_trials = 10
ss = 100
# # low rank problem
string ='final_blr_fullrank_results/result_blr_full_rank_problem'


example_1_data = ave_data(ss, 'notlazy_iaf', num_trials, string)
example_2_data = lazy_ave_data(500, 1, ss, 'lazy_iaf', num_trials, string)
example_3_data = lazy_ave_data(200, 3, ss, 'lazy_iaf', num_trials, string)


plt.figure()
plt.semilogy(example_1_data.steps, example_1_data.ave_neg_elbos, '*-', label='Baseline IAF')
plt.semilogy(example_2_data.steps, example_2_data.ave_neg_elbos,'o-', label='$U$-IAF')
plt.semilogy(example_3_data.steps, example_3_data.ave_neg_elbos,'x-', label='G3-IAF')
plt.legend(loc=0, fontsize = 18)
plt.xlabel('Iteration', fontsize=18)
plt.xticks([0, 5000, 10000, 15000, 20000], fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
fig_name = 'blr_fr_elbos.pdf'
plt.savefig(fig_name)
plt.show()


plt.figure()
plt.semilogy(example_1_data.steps, example_1_data.ave_traces,'*-', label='Baseline IAF')
plt.semilogy(example_2_data.steps, example_2_data.ave_traces,'o-', label='$U$-IAF')
plt.semilogy(example_3_data.steps, example_3_data.ave_traces,'x-', label='G3-IAF')
plt.legend(loc=0, fontsize = 18)
plt.xlabel('Iteration', fontsize=18)
plt.xticks([0, 5000, 10000, 15000, 20000], fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
fig_name = 'blr_fr_traces.pdf'
plt.savefig(fig_name)
plt.show()


plt.figure()
plt.semilogy(example_1_data.steps, example_1_data.ave_is_traces,'*-', label='Baseline IAF')
plt.semilogy(example_2_data.steps, example_2_data.ave_is_traces,'o-', label='$U$-IAF')
plt.semilogy(example_3_data.steps, example_3_data.ave_is_traces,'x-', label='G3-IAF')
plt.legend(loc=0, fontsize = 18)
plt.xlabel('Iteration', fontsize=18)
plt.xticks([0, 5000, 10000, 15000, 20000], fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
fig_name = 'blr_fr_is_traces.pdf'
plt.savefig(fig_name)
plt.show()



d1 = np.array(example_1_data.diagnostics)
d1[:,:2] = .5*d1[:,:2]  # 1/2 in trace diagnostic
d1[:,2] = -d1[:,2]  # flip ELBO to -ELBO
d1[:,3] = .5*d1[:,3]  # 1/2 in variance diagnostic

d2 = np.array(example_2_data.diagnostics)
d2[:,:2] = .5*d2[:,:2]  # 1/2 in trace diagnostic
d2[:,2] = -d2[:,2]  # flip ELBO to -ELBO
d2[:,3] = .5*d2[:,3]  # 1/2 in variance diagnostic

d3 = np.array(example_3_data.diagnostics)
d3[:,:2] = .5*d3[:,:2]  # 1/2 in trace diagnostic
d3[:,2] = -d3[:,2]  # flip ELBO to -ELBO
d3[:,3] = .5*d3[:,3]  # 1/2 in variance diagnostic

example_1_medians = np.median(d1, 0)
example_2_medians = np.median(d2, 0)
example_3_medians = np.median(d3, 0)

# shift ELBO to baseline
example_2_medians[2] = example_2_medians[2] - example_1_medians[2]
example_3_medians[2] = example_3_medians[2] - example_1_medians[2]


example_1_iqr = np.subtract(*np.percentile(d1, [75, 25], 0))
example_2_iqr = np.subtract(*np.percentile(d2, [75, 25], 0))
example_3_iqr = np.subtract(*np.percentile(d3, [75, 25], 0))

print('Tr(H^B), Tr(H), (Change in) ELBO , Variance diagnostic')
print('Baseline IAF', end=" ")
for m,s in zip(example_1_medians, example_1_iqr):
    print(' & $' + str(latex_float(m)) + '$ ($' + str(latex_float(s)) + '$)', end=" ")
print('\\\\')

print('$U$-IAF', end=" ")
for m, s in zip(example_2_medians, example_2_iqr):
    print(' & $' + str(latex_float(m)) + '$ ($' + str(latex_float(s)) + '$)', end=" ")
print('\\\\')

print('G3-IAF', end=" ")
for m,s in zip(example_3_medians, example_3_iqr):
    print(' & $' + str(latex_float(m)) + '$ ($' + str(latex_float(s)) + '$)', end=" ")
print('\\\\')

original_eigvals = example_3_data.eigvals[0][0]
after1_eigvals = example_3_data.eigvals[0][1]
after2_eigvals = example_3_data.eigvals[0][2]
after3_eigvals = example_3_data.eigvals[0][3]
plt.semilogy(original_eigvals[:100], '*-', label='Eig($H^{B}$)')
plt.semilogy(after1_eigvals[:100], 'o-', label='Eig($H^{B}_1$)')
plt.semilogy(after2_eigvals[:100], 'x-', label='Eig($H^{B}_2$)')
plt.semilogy(after3_eigvals[:100], '.-', label='Eig($H^{B}_3$)')
plt.xlabel('Eigenvalue index', fontsize=18)
plt.legend(loc=0, fontsize = 18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc=0, fontsize=18)
plt.tight_layout()
fig_name = 'blr_fr_evs.pdf'
plt.savefig(fig_name)
plt.show()
#
#
# idx = [i for i in range(1,21)]
# original_eigvals = example_3_data.eigvals[0][0]
# after1_eigvals = example_3_data.eigvals[0][1]
# plt.semilogy(idx, original_eigvals[:20], '*', label='eigenvalues of H')
# plt.semilogy(idx, after1_eigvals[:20], 'o', label='eigenvalues of H_1')
# plt.xlabel('Eigvalue index')
# plt.xticks(idx)
# plt.legend(fontsize = 14)
# plt.title('Eigenvalues of H matrices', fontsize = 16)
# fig_name = 'low_rank_eigvalues_after_training.png'
# plt.savefig(fig_name)
# plt.show()
