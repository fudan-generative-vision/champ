const videos = new Map<HTMLVideoElement, DOMRect>()

function playOrPause(video: HTMLVideoElement) {
  const { left, right, top, bottom } = video.getBoundingClientRect()
  if (bottom < 0 || top > window.innerHeight) {
    video.pause()
  } else if (left != 0 && right != 0) {
    video.play()
  }
}

const onscroll = (evt: Event) => {
  for (const video of videos.keys()) {
    playOrPause(video)
  }
}

export default {
  name: 'lazy',
  option: {
    mounted: (el: HTMLElement) => {
      if (el instanceof HTMLVideoElement) {
        videos.set(el, el.getBoundingClientRect())
        el.oncanplay = () => {
          videos.set(el, el.getBoundingClientRect())
          playOrPause(el)
        }
      }
      if (videos.size) {
        !window.onscroll && (window.onscroll = onscroll)
      }
    },
    unmounted: (el: HTMLElement) => {
      if (el instanceof HTMLVideoElement) {
        videos.delete(el)
      }
      if (!videos.size) {
        window.onscroll = null
      }
    }
  }
}
