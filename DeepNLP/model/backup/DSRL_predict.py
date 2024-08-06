def predict(self, eval_batch_size=16, data_path=None, sentence_list=None, verb_index_list=None):
        # no_cuda = not next(self.parameters()).is_cuda
        if data_path is not None:                                           # if provided with a dataset to be predicted
            eval_examples = self.load_data(data_path)
        elif sentence_list is not None and verb_index_list is not None:     # if provided with handmade examples
            eval_examples = self.load_data(sentence_list=sentence_list, verb_index_list=verb_index_list)
        language = get_language(''.join(eval_examples[0].text_a))
        label_map = {v: k for k, v in self.labelmap.items()}
        self.eval()
        all_pred = []

        for start_index in tqdm(range(0, len(eval_examples), eval_batch_size)):
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]
            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
            segment_ids, valid_ids = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                pred = self.forward(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                 verb_index=verb_index, labels=None,
                                 input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            lens = l_mask.sum(1).tolist()
            all_pred.extend(pred[l_mask].split(lens))

        label_map[0] = 'O'
        all_pred = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_pred]

        result_list = []

        # >-------仍需修改---------- #
        for pred, sentence in zip(all_pred, sentence_list):
            result_list.append([str(i)+'_'+str(j) for i, j in zip(sentence, pred)])
        # ----------------< #

        # print('write results to %s' % str(args.output_file))
        # with open(args.output_file, 'w', encoding='utf8') as writer:
        #     for i in range(len(y_pred)):
        #         sentence = eval_examples[i].text_a
        #         _, seg_pred_str = eval_sentence(y_pred[i], None, sentence, word2id)
        #         writer.write('%s\n' % seg_pred_str)

        return result_list