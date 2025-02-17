<h1>
  <img src="icon.png" alt="Speech-to-phrase logo" height="42" align="top">
  Speech-to-phrase
</h1>

A fast and local speech-to-text system that is personalized with your [Home Assistant](https://www.home-assistant.io/) device and area names.

Speech-to-phrase is not a general purpose speech recognition system. Instead of answering the question "what did the user say?", it answers "which of the phrases I know did the user say?".
This is accomplished by combining [pre-defined sentence templates](speech_to_phrase/sentences) with the names of your Home Assistant [entities, areas, and floors](https://www.home-assistant.io/getting-started/concepts-terminology/) that have been [exposed to Assist](https://www.home-assistant.io/voice_control/voice_remote_expose_devices/).

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=47701997_speech-to-phrase&repository_url=https%3A%2F%2Fgithub.com%2Frhasspy%2Fhassio-addons)

## Supported languages

- [English](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/english.md)
- [Français (French)](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/french.md)
- [Deutsch (German)](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/german.md)
- [Nederlands (Dutch)](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/dutch.md)
- [Spanish (Español)](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/spanish.md)
- [Italian (Italiano)](https://github.com/OHF-Voice/speech-to-phrase/blob/main/docs/italian.md)

## How it works

Speech-to-phrase combines [pre-defined sentence templates](speech_to_phrase/sentences) with the names of things from your Home Assistant to produce a [hassil](https://github.com/home-assistant/hassil) template file. This file compactly represents all of the possible sentences that can be recognized, which may be hundreds, thousands, or even millions.

Using techniques developed in the [Rhasspy project](https://rhasspy.readthedocs.io/en/latest/whitepaper/), speech-to-phrase converts the compact sentence templates into a [finite state transducer]((https://www.openfst.org)) (FST) which is then used to train a language model for [Kaldi](https://kaldi-asr.org/). The [opengrm](https://www.opengrm.org) tooling is crucial for efficiency during this step, as it avoids unpacking the sentence templates into every possible combination.

Each speech-to-phrase model contains a pre-built dictionary of word pronunciations as well as a [phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) model that will guess pronunciations for unknown words.

During training, a lot of "magic" happens to ensure that your entity, area, and floor names can be recognized automatically:

* Words with numbers are split apart ("PM2.5" becomes "PM 2.5")
* Initialisms are further split ("PM" or "P.M." becomes "P M")
* Digits are replaced with their [spoken word forms](https://github.com/rhasspy/unicode-rbnf) ("123" becomes "one hundred twenty three")
* Unknown words have their pronunciations guessed

To make phrase recognition more robust, a "fuzzy" layer is added on top of Kaldi's transcription output. This layer can correct small errors, such as duplicate or missing words, and also ensures that output names are exactly what you have in Home Assistant.

## Models and tools

Speech models and tools are downloaded automatically from [HuggingFace](https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main)

