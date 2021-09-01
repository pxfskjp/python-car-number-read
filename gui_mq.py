import pika
import os, sys

mq_user = "lisc"
mq_passwd="8J7Sk4J3jYsBhxfJ"
mq_host = "license_scanner"
mq_port="5672"
mq_channel="plates"
mq_dns="rabbitmq-1.demolium.eu"
mq_url = "amqp://%s:%s@rabbitmq-1.demolium.eu:%s/%s" % (mq_user, mq_passwd, mq_port, mq_host)
mq_connection=""

def sendmessage(carplatenumber):
    #params = pika.URLParameters(mq_url)
    connection = pika.BlockingConnection(pika.ConnectionParameters(mq_dns, int(mq_port), mq_host, pika.PlainCredentials(mq_user, mq_passwd)))
    channel = connection.channel()  # start a channel
    channel.queue_declare(queue=mq_channel, durable=True)  # Declare a queue
    channel.basic_publish(exchange='',
                          routing_key=mq_channel,
                          body=carplatenumber)

    print(" [x] Sent '"+ carplatenumber+ "'")
    connection.close()


def receiveMessage(_callback):
    global mq_connection
    try:

        mq_connection = pika.BlockingConnection(pika.ConnectionParameters(mq_dns, int(mq_port), mq_host, pika.PlainCredentials(mq_user, mq_passwd)))
        channel = mq_connection.channel()
        channel.queue_declare(queue=mq_channel, durable=True)
        channel.basic_consume(queue=mq_channel, on_message_callback=_callback, auto_ack=True)
        channel.basic_qos(prefetch_count=1)
        channel.start_consuming()
    except:
        print("ERROR IN CONNECTIng and fetching messages")
        pass


def callback(ch, method, properties, body):
    print("came in callback function")
    print(" [x] Received %r" % body)
    ch.close()
    #if mq_connection!="":
    #    mq_connection.close()

def main():
    if sys.argv[1]=='1':
        for i in range(2):
            sendmessage( str(i) )
    else:
        receiveMessage(callback)
        
#main()
