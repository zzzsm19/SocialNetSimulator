def get_prompt(prompt_template_path, prompt_input):
    with open(prompt_template_path, 'r', encoding='gbk') as f:
        prompt = f.read()
    if "<---------->" in prompt:
        prompt = prompt.split("<---------->")[1].strip()
    for key, value in prompt_input.items():
        prompt = prompt.replace("{<" + key + ">}", str(value))
    return prompt