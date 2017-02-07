#ifndef FOC_FEATURE_H
#define FOC_FEATURE_H

void foc_feature_extraction_init(std::vector<FOC_Feature_t> &data_feature);
bool foc_feature_extraction_update(std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2], std::vector<FOC_Feature_t> &data_feature, int size_of_signal);

#endif
