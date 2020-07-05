model <- transformer(architecture = "FlauBERT", model_name = "flaubert/flaubert_small_cased")
expect_true(inherits(model, "Transformer"))
expect_true(inherits(predict(model, "testing out", type = "embed-sentence", trace = FALSE), "matrix"))

## Clean up models folder
unlink_golgotha()
