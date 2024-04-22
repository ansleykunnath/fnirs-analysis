import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

ch_of_interest = 'S10_D18 hbo'
#ch_of_interest = 'S17_D13 hbo'
#ch_of_interest = 'S14_D26 hbo'

############ Plot Trained Pre ############
rdm_dict_y = trained_rdms_pre[ch_of_interest]
rdm_dict_x = trained_rdms_pre[ch_of_interest]

# Initialize the RDM matrix
num_conditions_x = len(rdm_dict_x)
num_conditions_y = len(rdm_dict_y)
rdm_matrix_trained_pre = np.zeros((num_conditions_y, num_conditions_x))

# Fill in the RDM matrix
for i, (cond_y, dissimilarity_list_y) in enumerate(rdm_dict_y.items()):
    for j, (cond_x, dissimilarity_list_x) in enumerate(rdm_dict_x.items()):
        # Use dissimilarities from both dictionaries for the RDM
        rdm_matrix_trained_pre[i, j] = np.mean(np.abs(np.array(dissimilarity_list_y) - np.array(dissimilarity_list_x)))

# Plotting the RDM
plt.imshow(rdm_matrix_trained_pre, cmap='viridis', interpolation='nearest')
plt.clim(0.0000000,0.0000004) 
plt.colorbar(label='Dissimilarity')
plt.yticks(range(num_conditions_y), list(rdm_dict_y.keys()))
plt.xticks(range(num_conditions_x), list(rdm_dict_x.keys()), rotation=45)
plt.title('Trained - Day 1')
plt.ylabel('Conditions (Before)')
plt.xlabel('Conditions (After)')
plt.show()


############ Plot Trained Post ############
rdm_dict_y = trained_rdms_post[ch_of_interest]
rdm_dict_x = trained_rdms_post[ch_of_interest]

# Initialize the RDM matrix
num_conditions_x = len(rdm_dict_x)
num_conditions_y = len(rdm_dict_y)
rdm_matrix_trained_post = np.zeros((num_conditions_y, num_conditions_x))

# Fill in the RDM matrix
for i, (cond_y, dissimilarity_list_y) in enumerate(rdm_dict_y.items()):
    for j, (cond_x, dissimilarity_list_x) in enumerate(rdm_dict_x.items()):
        # Use dissimilarities from both dictionaries for the RDM
        rdm_matrix_trained_post[i, j] = np.mean(np.abs(np.array(dissimilarity_list_y) - np.array(dissimilarity_list_x)))

# Plotting the RDM
plt.imshow(rdm_matrix_trained_post, cmap='viridis', interpolation='nearest')
plt.clim(0.0000000,0.0000004) 
plt.colorbar(label='Dissimilarity')
plt.yticks(range(num_conditions_y), list(rdm_dict_y.keys()))
plt.xticks(range(num_conditions_x), list(rdm_dict_x.keys()), rotation=45)
plt.title('Trained - Day 3')
plt.ylabel('Conditions')
plt.xlabel('Conditions')
plt.show()



############ Plot Control Pre ############
rdm_dict_y = control_rdms_pre[ch_of_interest]
rdm_dict_x = control_rdms_pre[ch_of_interest]

# Initialize the RDM matrix
num_conditions_x = len(rdm_dict_x)
num_conditions_y = len(rdm_dict_y)
rdm_matrix_control_pre = np.zeros((num_conditions_y, num_conditions_x))

## Fill in the RDM matrix
for i, (cond_y, dissimilarity_list_y) in enumerate(rdm_dict_y.items()):
    for j, (cond_x, dissimilarity_list_x) in enumerate(rdm_dict_x.items()):
        # Use dissimilarities from both dictionaries for the RDM
        rdm_matrix_control_pre[i, j] = np.mean(np.abs(np.array(dissimilarity_list_y) - np.array(dissimilarity_list_x)))

# Plotting the RDM
plt.imshow(rdm_matrix_control_pre, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Dissimilarity')
plt.clim(0.0000000,0.0000007) 
plt.yticks(range(num_conditions_y), list(rdm_dict_y.keys()))
plt.xticks(range(num_conditions_x), list(rdm_dict_x.keys()), rotation=45)
plt.title('Control - Day 1')
plt.ylabel('Conditions')
plt.xlabel('Conditions')
plt.show()


############ Plot Control Post ############
rdm_dict_y = control_rdms_post[ch_of_interest]
rdm_dict_x = control_rdms_post[ch_of_interest]

# Initialize the RDM matrix
num_conditions_x = len(rdm_dict_x)
num_conditions_y = len(rdm_dict_y)
rdm_matrix_control_post = np.zeros((num_conditions_y, num_conditions_x))

# Fill in the RDM matrix
for i, (cond_y, dissimilarity_list_y) in enumerate(rdm_dict_y.items()):
    for j, (cond_x, dissimilarity_list_x) in enumerate(rdm_dict_x.items()):
        # Use dissimilarities from both dictionaries for the RDM
        rdm_matrix_control_post[i, j] = np.mean(np.abs(np.array(dissimilarity_list_y) - np.array(dissimilarity_list_x)))

# Plotting the RDM
plt.imshow(rdm_matrix_control_post, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Dissimilarity')
plt.clim(0.0000000,0.0000008) 
plt.yticks(range(num_conditions_y), list(rdm_dict_y.keys()))
plt.xticks(range(num_conditions_x), list(rdm_dict_x.keys()), rotation=45)
plt.title('Control - Day 7')
plt.ylabel('Conditions')
plt.xlabel('Conditions')
plt.show()

