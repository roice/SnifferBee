#ifndef FOC_WAVELET_H
#define FOC_WAVELET_H

void foc_cwt_init(float *data_wvs, std::vector<int>& data_wvs_idx, std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS]);
bool foc_cwt_update(std::vector<float> *signal, std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS]);

void foc_identify_modmax_init(std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2]);
bool foc_identify_modmax_update(std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS], std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2]);

void foc_chain_maxline_init(std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2]);
bool foc_chain_maxline_update(std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2], std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2], int size_of_wt_out);

#endif
