
def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """
    Align labels to the tokenized word, special token like [CLS], [SEP] will be assigned -100
    Return a dict with the aligned labels
    """

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_and_chunk(example, tokenizer, stride=0):
    """
    Will return a dict with aligned label lists
    {
        'input_ids'      : a list of length "chunk_num"
        'attention_mask' : a list of length "chunk_num"
        'labels'         : a list of length "chunk_num"
    }
    """

    tokenized_input_chunk = tokenizer(
        example['tokens'],
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=stride,
        is_split_into_words=True
    )

    chunk_num = len(tokenized_input_chunk['overflow_to_sample_mapping'])

    label = example['ner_tags']
    labels = []

    for i in range(chunk_num):
        word_ids = tokenized_input_chunk.word_ids(i)
        previous_word_idx = None
        label_ids = []
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(-100)
            # if still in the overlapping scope, set to invalid labels
            elif i > 0 and j <= stride:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
        
        
        labels.append(label_ids)

    tokenized_input_chunk['labels'] = labels
    tokenized_input_chunk.pop('overflow_to_sample_mapping')
    tokenized_input_chunk['chunk_num'] = chunk_num
    
    return tokenized_input_chunk
