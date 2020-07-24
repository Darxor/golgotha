model <- transformer(architecture = "BART", model_name = "sshleifer/distilbart-xsum-12-1")
expect_true(inherits(model, "Transformer"))
expect_true(inherits(predict(model, " For any pretrained model the quality of the “pre-training” process is critical. The Facebook researchers used a dataset extracted from the common crawl corpus of 25 languages (CC25) as a subset and balanced with up/down-sampling based on the percentage for each language in CC25.",
                             type = "embed-sentence", trace = TRUE), "matrix"))
#TODO BUG  Error in py_call_impl(callable, dots$args, dots$keywords) :
# AssertionError: BartModel(
#   (shared): Embedding(50264, 1024, padding_idx=1)...)  should have a 'get_encoder' function defined",

expect_true(inherits(predict(model, newdata=" If I was poor, I would ", type = "generate",  trace = TRUE), "matrix"))

## Clean up models folder
unlink_golgotha()
