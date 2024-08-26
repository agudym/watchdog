from typing import Optional, Dict, List
import io, re, requests, logging

import numpy as np

class TeleBotComm:
    """
    Simple telegram bot wrapper for messages retrieving/sending, images upload 
    """
    def __init__(self, bot_token:str, chat_id:Optional[int]=None):
        self._url_get_message = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        self._url_send_message = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._url_send_image = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

        self.chat_id = chat_id
        self.last_message_id = -1

        # Check the latest message id
        self.get_message(strict=True)

    def get_message(self, is_latest_msg:bool=True, strict:bool=False):
        """ Parse messages, return empty string in case of connection errors """
        url = self._url_get_message
        if is_latest_msg:
            url += "?offset=-1"

        try:
            ret_json = requests.post(url).json()
        except requests.exceptions.ConnectionError as e:
            if strict:
                raise
            logging.error(e)
            return ""

        if self._check_ret_json(ret_json) :
            for response in ret_json["result"]:
                if "message" not in response:
                    continue

                msg = response["message"]
                msg_chat_id = msg["from"]["id"]
                msg_id = msg["message_id"]
                
                if is_latest_msg and self.last_message_id >= msg_id:
                    continue

                if self.chat_id is None:
                    self.chat_id = msg_chat_id

                if self.chat_id == msg_chat_id:
                    self.last_message_id = msg_id
                    logging.info(f"Message from chat {msg_chat_id}")
                    return msg["text"]
        return ""

    def send_message(self, msg:str) -> bool:
        """ Send message (return True) or return False in case of connection issues """
        if self.chat_id is None:
            raise ValueError("Chat ID is not initialized.")

        try:
            ret_json = requests.post(
                self._url_send_message, json={'chat_id': self.chat_id, 'text': msg}).json()
        except requests.exceptions.ConnectionError as e:
            logging.error(e)
            return False

        return self._check_ret_json(ret_json)

    def send_image(self, image_file:io.BufferedIOBase, title:str="") -> bool:
        """ Send image (return True) or return False in case of connection issues  """
        try:
            ret_json = requests.post(
                self._url_send_image,
                data={"chat_id": self.chat_id, "caption": title},
                files={"photo": image_file}).json()
        except requests.exceptions.ConnectionError as e:
            logging.error(e)
            return False
        
        return self._check_ret_json(ret_json)

    def _check_ret_json(self, ret_json: Dict) -> bool:
        if ret_json["ok"] :
            return True
        else:
            logging.error(f"Request error: {ret_json['error_code']}")
            return False

class CameraTeleBotComm(TeleBotComm):
    """
    Extended `TeleBotComm` for specific commands
    """
    def __init__(self,
                 warning_timeout:int,
                 detect_conf_thr:float,
                 categories_all:List[str],
                 categories_notify:List[str],
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._args_desc = {}
        self.status = False
        self._args_desc["S"] = "Flag to get current system's status (no value)"

        self.exit = False
        self._args_desc["X"] = "Flag to stop the watchdog (no value), the operation is irreversible"

        self.warning_timeout = warning_timeout
        self._args_desc["T"] = "Minimal interval between the warning messages [seconds], e.g. 'T 30'"
        
        self.detect_conf_thr = detect_conf_thr
        self._args_desc["P"] = "Detection confidence/probability threshold [0.1 ... 1], e.g. 'P 0.5', or 'P 1' to disable all notifications"

        self.categories_all = categories_all
        self.categories_notify = categories_notify
        self._args_desc["O"] = "List of object categories to detect, e.g. 'O person dog cat bird'"

        self._help = "Watchdog bot usage:\n<parameter> [value]\nParameters:\n"
        for arg_name, arg_desc in self._args_desc.items():
            self._help += f"{arg_name}: {arg_desc}\n"
        self._help += "'H' for help."
        
        if not self.send_message(self._help) :
            raise RuntimeError("CameraTeleBotComm initialization failure!")
    
    def parse(self) -> bool:
        """
        Parse telegram bot's input, return True in case of any update.
        Timeout must between requests must be set manually, e.g. time.sleep(1)!
        """
        msg = self.get_message()
        if len(msg) == 0:
            return False

        msg_split = msg.split()
        if len(msg_split[0]) == 0:
            return False
        
        parameter_name = msg_split[0]
        if parameter_name == "H":
            self.send_message(self._help)
            return False

        if parameter_name not in self._args_desc.keys() :
            estr = f"Uknown parameter {parameter_name}! Send 'H' for help."
            logging.warning(estr)
            self.send_message(estr)
            return False

        value = None
        if parameter_name == "S":
            value = True
            self.status = value
        elif parameter_name == "X":
            value = True
            self.exit = value
        elif len(msg_split) >= 2:
            if parameter_name == "T":
                value = np.clip(float(msg_split[1]), 0, 3600 * 24) # snooze for a day
                self.warning_timeout = value
            elif parameter_name == "P":
                value = np.clip(float(msg_split[1]), 0, 1)
                self.detect_conf_thr = value
            else:# parameter_name == "O":
                value = re.split("\W+", msg)
                if len(value) <= 1:
                    estr = f"Invalid object names, possible format is 'O person dog cat bird' !"
                    logging.warning(estr)
                    self.send_message(estr)
                    return False
                value = value[1:]
                if not np.all(np.isin(value, self.categories_all)):
                    estr = f"Invalid object category, use values from the following list: {' '.join(self.categories_all)}"
                    logging.warning(estr)
                    self.send_message(estr)
                    return False

                self.categories_notify = value
        else:
            estr = f"Invalid parameter {parameter_name} value! Send 'H' for help."
            logging.warning(estr)
            self.send_message(estr)
            return False

        estr = f"Parameter '{parameter_name}' set to '{value}'"
        logging.info(estr)

        return self.send_message(estr)

if __name__ == "__main__":
    import argparse, time

    from .utils import setup_logger, load_config, save_config
    setup_logger(verbose=True)

    parser = argparse.ArgumentParser(description="Verify Telegram bot communication (with TOKEN from Telegram's '@BotFather'): acquires chat-id, updates config, and checks communication!")
    parser.add_argument("config_json_path", type=str, help="Path to the main config file with the TOKEN")
    args = parser.parse_args()

    config_dict = load_config(args.config_json_path)
    token = config_dict["Bot"]["token"]
    chat_id_str = config_dict["Bot"]["chat_id"]
    category_names_all = config_dict["Detector"]["categories_all"].split(",")
    category_names_notify = config_dict["Detector"]["categories_notify"].split(",")

    if token == "":
        raise ValueError("'token' is not set in the config!")

    if len(chat_id_str) == 0:
        # Logging chat id
        bot = TeleBotComm(token)
        if bot.chat_id is None :
            raise RuntimeError("No active user messages found, unable to get chat-id. Text something to the bot and restart!")
        chat_id_str = str(bot.chat_id)
        config_dict["Bot"]["chat_id"] = chat_id_str
        save_config(args.config_json_path, config_dict)
        print(f"Chat-id updated in {args.config_json_path}.")

    bot = CameraTeleBotComm(0,0.99, category_names_all, category_names_notify, token, int(chat_id_str))
    print("Bot is ready. Check your Telegram...")
    if not bot.send_message("Try to set some parameter..."):
        raise RuntimeError("Communication failure!")

    while True:
        if bot.parse() and bot.send_message("Bot test-run ends..."):
            print("Communication works!")
            break
        time.sleep(1)
