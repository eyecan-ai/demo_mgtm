import signal
import click
from orione.interfaces.relate.server import RelateServer


@click.command("Start inference server")
@click.option("--host", default='localhost', help="Host")
@click.option("--port", default=44441, type=int, help="Port")
@click.option("--checkpoint_path", required=True, help="Saved checkpoint path")
@click.option("--decoder_cfg", required=True, help="Decoder configuration path")
@click.option("--gpu/--cpu", default=True, help="Use CUDA gpus")
def inference(host, port, checkpoint_path, decoder_cfg, gpu):

    # Create server
    server = RelateServer(
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        decoder_cfg=decoder_cfg,
        gpu=gpu
    )
    server.start()

    print('Press CTRL+C to gracefully shutdown the server')
    signal.signal(signal.SIGINT, lambda signum, frame: server.stop())
    signal.pause()


if __name__ == "__main__":
    inference()
