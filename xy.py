from wxpy import *
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.utils import EndpointConfig

					
agent = Agent('w_domain.yml', policies = [MemoizationPolicy(), KerasPolicy(max_history=3, epochs=200, batch_size=50)])
data = agent.load_data('./data/stories.md')	
agent.train(data)
interpreter = RasaNLUInterpreter('./models/nlu/default/trainedNlu')
action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
agent = Agent.load('./models/dialogue', interpreter=interpreter, action_endpoint=action_endpoint)
    

bot = Bot(console_qr=False, cache_path=True)
@bot.register(bot.friends())
def reply_my_friend(msg):
    ans = agent.handle_message(msg.text)
    print(ans)
    return ans[0]['text']

embed()