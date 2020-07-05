model <- transformer(architecture = "CamenBERT", model_name = "Musixmatch/umberto-commoncrawl-cased-v1")
expect_true(inherits(model, "Transformer"))
expect_true(inherits(predict(model, "testing out", type = "embed-sentence", trace = FALSE), "matrix"))

## Clean up models folder
unlink_golgotha()
