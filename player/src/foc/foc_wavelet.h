#ifndef FOC_WAVELET_H
#define FOC_WAVELET_H

void foc_cwt_init(float **addr_data_wvs, std::vector<int>& data_wvs_idx, float **data_wt_out, std::vector<int>& data_wt_idx);
bool foc_cwt_update(std::vector<float> *signal, float **data_wt_out, std::vector<int>& data_wt_idx);

#endif
