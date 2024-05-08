#include <iostream>
#include <fstream>
#include <vector>

static std::vector<std::vector<float>> get_std_distr(std::vector<std::vector<float>>* data) {

	std::vector<std::vector<float>> mean_dev((*data)[0].size(), std::vector<float>(2));

	for (int _data_dimension = 0; _data_dimension < (*data)[0].size(); _data_dimension++) {
		for (int _data_point = 0; _data_point < (*data).size(); _data_point++) {
			mean_dev[_data_dimension][0] += (*data)[_data_point][_data_dimension];
		}
		mean_dev[_data_dimension][0] /= (float)(*data).size();
	}

	float temp_square;
	for (int _data_dimension = 0; _data_dimension < (*data)[0].size(); _data_dimension++) {
		for (int _data_point = 0; _data_point < (*data).size(); _data_point++) {
			temp_square = (*data)[_data_point][_data_dimension] - mean_dev[_data_dimension][0];
			mean_dev[_data_dimension][1] += temp_square * temp_square;
		}
		mean_dev[_data_dimension][1] /= (float)(*data).size();
		mean_dev[_data_dimension][1] = sqrt(mean_dev[_data_dimension][1]);
	}

	return mean_dev;
}

static void write_standardized_data(std::ofstream* output_file, std::vector<std::vector<float>>* processed_data,
	std::vector<std::vector<float>>* std_distr) {

	for (int _data_point = 0; _data_point < (*processed_data).size(); _data_point++) {
		for (int _component = 0; _component < (*processed_data)[0].size() - 1; _component++) {
			(*output_file) << ((*processed_data)[_data_point][_component] - (*std_distr)[_component][0])
				/ (*std_distr)[_component][1] << ";";
		}
		(*output_file) << ((*processed_data)[_data_point][(*processed_data)[0].size() - 1]
			- (*std_distr)[(*processed_data)[0].size() - 1][0]) / (*std_distr)[(*processed_data)[0].size() - 1][1] << "\n";
	}
}

static std::vector<std::vector<float>> standard_mask_processing(std::ofstream* output_file,
	std::vector<uint8_t>* data, std::vector<std::vector<float>>* masks,
	int dim_data, bool is_training, std::vector<std::vector<float>>* in_std_distr) {

	int data_iterator = 0;
	int current_id_sum_int = 0;
	float current_convolve = 0.0;
	float current_id_sum_float = 0.0;
	std::vector<std::vector<float>> processed_data((*data).size() / dim_data, std::vector<float>((*masks).size()));

	for (int _data_point = 0; _data_point < (*data).size() / dim_data; _data_point++) {

		current_id_sum_int = 0;
		for (int _component = 0; _component < dim_data; _component++) {
			current_id_sum_int += (*data)[data_iterator];
			data_iterator++;
		}
		data_iterator -= dim_data;
		current_id_sum_float = (float)current_id_sum_int;

		for (int _mask = 0; _mask < (*masks).size(); _mask++) {
			current_convolve = 0.0;
			for (int _component = 0; _component < dim_data; _component++) {
				current_convolve += (float)(*data)[data_iterator] * (*masks)[_mask][_component];
				data_iterator++;
			}
			processed_data[_data_point][_mask] = current_convolve / current_id_sum_float;
			data_iterator -= dim_data;
		}
		data_iterator += dim_data;
	}

	if (is_training) {
		std::vector<std::vector<float>> std_distr = get_std_distr(&processed_data);
		write_standardized_data(output_file, &processed_data, &std_distr);
		return std_distr;
	}

	write_standardized_data(output_file, &processed_data, in_std_distr);
	return (*in_std_distr);
}

static std::vector<std::vector<float>> standard_differential_processing(std::ofstream* output_file,
	std::vector<uint8_t>* data, int side_len, bool is_training,
	std::vector<std::vector<float>>* in_std_distr) {

	std::vector<std::vector<float>> differential_data((*data).size() / side_len / side_len,
		std::vector<float>(side_len*2));

	int scaled_y;

	for (int _data_point = 0; _data_point < differential_data.size(); _data_point++) {
		scaled_y = _data_point * side_len * side_len;
		for (int _y_component = 0; _y_component < side_len; _y_component++) {
			for (int _x_component = 0; _x_component < side_len - 1; _x_component++) {
				if (((*data)[scaled_y + _x_component + 1] == 0) xor
					((*data)[scaled_y + _x_component] == 0)) {
					differential_data[_data_point][_y_component + side_len]++;
				}
			}
			scaled_y += side_len;
		}
	}

	for (int _data_point = 0; _data_point < differential_data.size(); _data_point++) {
		for (int _x_component = 0; _x_component < side_len; _x_component++) {
			scaled_y = _data_point * side_len * side_len;
			for (int _y_component = 0; _y_component < side_len - 1; _y_component++) {
				if (((*data)[scaled_y + _x_component + side_len] == 0) xor
					((*data)[scaled_y + _x_component] == 0)) {
					differential_data[_data_point][_x_component]++;
				}
				scaled_y += side_len;
			}
		}
	}

	if (is_training) {
		std::vector<std::vector<float>> std_distr = get_std_distr(&differential_data);
		write_standardized_data(output_file, &differential_data, &std_distr);
		return std_distr;
	}

	write_standardized_data(output_file, &differential_data, in_std_distr);
	return (*in_std_distr);
}

static std::vector<std::vector<float>> gen_digit_masks(std::vector<uint8_t>* data,
	std::vector<uint8_t>* target, int dim_data, int size_target) {

	std::vector<std::vector<float>> masks(size_target, std::vector<float>(dim_data));

	int current_label = 0;
	int data_iterator = 0;
	for (int _data_point = 0; _data_point < (*target).size(); _data_point++) {
		current_label = (*target)[_data_point];
		for (int _component = 0; _component < dim_data; _component++) {
			masks[current_label][_component] += (float)(*data)[data_iterator];
			data_iterator++;
		}
	}

	std::vector<float> masks_id_sum(size_target, 0.0);

	for (int _mask = 0; _mask < size_target; _mask++) {
		for (int _component = 0; _component < dim_data; _component++) {
			masks_id_sum[_mask] += masks[_mask][_component];
		}
		for (int _component = 0; _component < dim_data; _component++) {
			masks[_mask][_component] /= masks_id_sum[_mask];
		}
	}

	return masks;
}


int main() {

	// loading data

	constexpr int size_target_header = 8;
	constexpr int size_data_header = 16;
	constexpr int size_target = 10;

	std::ifstream input_train_data("train-images-idx3-ubyte", std::ios::binary);
	if (!input_train_data.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<uint8_t> train_data(std::istreambuf_iterator<char>(input_train_data), {});
	input_train_data.close();
	train_data.erase(train_data.begin(), std::next(train_data.begin(), size_data_header));

	std::ifstream input_test_data("t10k-images-idx3-ubyte", std::ios::binary);
	if (!input_test_data.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<uint8_t> test_data(std::istreambuf_iterator<char>(input_test_data), {});
	input_test_data.close();
	test_data.erase(test_data.begin(), std::next(test_data.begin(), size_data_header));

	std::ifstream input_labels("train-labels-idx1-ubyte", std::ios::binary);
	if (!input_labels.is_open()) {
		std::cout << "unable to open source file";
		return -1;
	}
	std::vector<uint8_t> target(std::istreambuf_iterator<char>(input_labels), {});
	input_labels.close();
	target.erase(target.begin(), std::next(target.begin(), size_target_header));
	int dim_data = train_data.size() / target.size();

	// data processing

	std::ofstream output_diff_train_data("differential_train_data.txt");
	std::vector<std::vector<float>> diff_train_std_distr(56, std::vector<float>(2, 1.0));
	/*for (int i = 0; i < 56; i++) {
		diff_train_std_distr[i][0] = 0.0;
	}
	standard_differential_processing(&output_diff_train_data, &train_data, 28, false, &diff_train_std_distr); */
	diff_train_std_distr = standard_differential_processing(&output_diff_train_data, &train_data, 28, true, {});
	output_diff_train_data.close();

	std::ofstream output_diff_test_data("differential_test_data.txt");
	standard_differential_processing(&output_diff_test_data, &test_data, 28, false, &diff_train_std_distr);
	output_diff_test_data.close();

	std::vector<std::vector<float>> digit_masks(size_target, std::vector<float>(dim_data));
	digit_masks = gen_digit_masks(&train_data, &target, dim_data, size_target);


	std::ofstream output_mask_train_data("mean_digit_convolution_train_data.txt");
	std::vector<std::vector<float>> mask_train_std_distr;
	mask_train_std_distr = standard_mask_processing(&output_mask_train_data, &train_data, &digit_masks, dim_data, true, {});
	output_mask_train_data.close();

	std::ofstream output_mask_test_data("mean_digit_convolution_test_data.txt");
	standard_mask_processing(&output_mask_test_data, &test_data, &digit_masks, dim_data, false, &mask_train_std_distr);
	output_mask_test_data.close();

	return 0;
}