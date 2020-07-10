model <- transformer(architecture = "T5", model_name = "t5-small")
expect_true(inherits(model, "Transformer"))
# TODO Fails with  Error in py_call_impl(callable, dots$args, dots$keywords) :
# ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
expect_true(inherits(predict(model, "testing out", type = "embed-sentence", trace = TRUE), "matrix"))

## Clean up models folder
unlink_golgotha()
