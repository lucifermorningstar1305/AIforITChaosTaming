from typing import Dict, Callable, List, Tuple

import numpy as np
import polars as pol
import torch
import torch.utils.data as td


class SupportTicketDataset(td.Dataset):
    def __init__(
        self,
        data: pol.DataFrame,
        tokenizer: Callable,
        is_bert_based: bool = True,
        context_length: int = 512,
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.is_bert_based = is_bert_based
        self.context_length = context_length

    def __len__(self):
        return self.data.shape[0]

    def _tokenize_text(
        self,
        text: str,
        do_truncate: bool = True,
        do_pad: bool = True,
        add_special_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to tokenize the text data."""

        padding = "max_length" if do_pad else "do_not_pad"

        tokenized_results = self.tokenizer(
            text,
            truncation=do_truncate,
            padding=padding,
            max_length=self.context_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

        return (
            tokenized_results["input_ids"].reshape(-1),
            tokenized_results["attention_mask"].reshape(-1),
        )

    def _split_overlapping_tokens(
        self,
        tensor: torch.Tensor,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
    ) -> List:
        """Function to split the long texts into overlapping chunks for context length matching."""
        results = [tensor[i : i + chunk_size] for i in range(0, tensor.size(0), stride)]

        if len(results) > 1:
            results = list(filter(lambda x: x.size(0) >= minimal_chunk_length, results))

        return results

    def _add_special_tokens(
        self, tensor_chunks: List[torch.Tensor], mask_chunks: List[torch.Tensor]
    ) -> Tuple[list, list]:
        """Function to add special tokens at the begining or the end or both."""

        if self.is_bert_based:
            for i in range(len(tensor_chunks)):
                tensor_chunks[i] = torch.cat(
                    [torch.Tensor([101]), tensor_chunks[i], torch.Tensor([102])]
                )

                mask_chunks[i] = torch.cat(
                    [torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])]
                )

        return tensor_chunks, mask_chunks

    def _add_padding_tokens(
        self, tensor_chunks: List[torch.Tensor], mask_chunks: List[torch.Tensor]
    ) -> Tuple[list, list]:
        """Function to add the padding tokens."""

        if self.is_bert_based:
            for i in range(len(tensor_chunks)):
                pad_len = self.context_length - tensor_chunks[i].size(-1)
                print(
                    f"Context length: {self.context_length}, Input length: {tensor_chunks[i].size(-1)}, Pad length: {pad_len}"
                )

                if pad_len > 0:
                    tensor_chunks[i] = torch.cat(
                        [tensor_chunks[i], torch.Tensor([0] * pad_len)]
                    )

                    mask_chunks[i] = torch.cat(
                        [mask_chunks[i], torch.Tensor([0] * pad_len)]
                    )

        return tensor_chunks, mask_chunks

    def _stack_all_tensors(
        self, tensor_chunks: List[torch.Tensor], mask_chunks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to stack all the chunks into a single tensor."""

        tensor_chunks = torch.stack(tensor_chunks)
        mask_chunks = torch.stack(mask_chunks)

        return tensor_chunks.long(), mask_chunks.int()

    def __getitem__(self, index) -> Dict:

        clean_text = self.data.item(index, "clean_text")
        label = self.data.item(index, "label")
        input_ids, attention_mask = self._tokenize_text(
            text=clean_text,
            do_truncate=False,
            do_pad=False,
            add_special_tokens=False,
        )

        token_len = input_ids.size(0)

        if token_len >= self.context_length:

            input_ids_chunks = self._split_overlapping_tokens(
                tensor=input_ids, chunk_size=510, stride=510, minimal_chunk_length=1
            )

            attention_mask_chunks = self._split_overlapping_tokens(
                tensor=attention_mask,
                chunk_size=510,
                stride=510,
                minimal_chunk_length=1,
            )

            assert all(
                [x.size(-1) <= 512 for x in input_ids_chunks]
            ), "Found chunks in the input_ids chunks where the length is greater than the context length"

            assert all(
                [x.size(-1) <= 512 for x in attention_mask_chunks]
            ), "Found chunks in the attention_mask chunks where the length is greater than the context length"

            input_ids_chunks, attention_mask_chunks = self._add_special_tokens(
                tensor_chunks=input_ids_chunks, mask_chunks=attention_mask_chunks
            )

            input_ids_chunks, attention_mask_chunks = self._add_padding_tokens(
                tensor_chunks=input_ids_chunks, mask_chunks=attention_mask_chunks
            )

            input_ids, attention_mask = self._stack_all_tensors(
                tensor_chunks=input_ids_chunks, mask_chunks=attention_mask_chunks
            )

            assert (
                input_ids.size(-1) <= 512
            ), f"Expected input_ids to have context length : {self.context_length}. Found {input_ids.size()} -- chunk section"

            assert (
                input_ids.size(-1) <= 512
            ), f"Expected attention_mask to have context length : {self.context_length}. Found {attention_mask.size()} -- chunk section"

        else:

            input_ids, attention_mask = self._add_special_tokens(
                tensor_chunks=[input_ids], mask_chunks=[attention_mask]
            )
            print("Before Pad: ", input_ids[0].shape)

            input_ids, attention_mask = self._add_padding_tokens(
                tensor_chunks=input_ids, mask_chunks=attention_mask
            )
            print("After Pad: ", input_ids[0].shape)

            input_ids, attention_mask = self._stack_all_tensors(
                tensor_chunks=input_ids, mask_chunks=attention_mask
            )

            if input_ids.size(-1) > 512:
                print(clean_text)

            assert (
                input_ids.size(-1) <= 512
            ), f"Expected input_ids to have context length : {self.context_length}. Found {input_ids.size()} -- token_len"

            assert (
                attention_mask.size(-1) <= 512
            ), f"Expected attention_mask to have context length : {self.context_length}. Found {attention_mask.size()} -- token_len"

        return input_ids, attention_mask, torch.tensor(label).long()
