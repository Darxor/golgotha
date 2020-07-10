model <- transformer(architecture = "BART", model_name = "sshleifer/distilbart-xsum-12-1")
expect_true(inherits(model, "Transformer"))
expect_true(inherits(predict(model, "For any pretrained model the quality of the “pre-training” process is critical. The Facebook researchers used a dataset extracted from the common crawl corpus of 25 languages (CC25) as a subset and balanced with up/down-sampling based on the percentage for each language in CC25.",
                             type = "embed-sentence", trace = TRUE), "matrix"))
expect_true(inherits(predict(model, "For any pretrained model the quality of the “pre-training” process is critical. The Facebook researchers used a dataset extracted from the common crawl corpus of 25 languages (CC25) as a subset and balanced with up/down-sampling based on the percentage for each language in CC25.",
                             trace = TRUE), "matrix"))

## Clean up models folder
unlink_golgotha()
