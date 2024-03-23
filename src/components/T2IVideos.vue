<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { initTWE, Carousel } from 'tw-elements';
onMounted(() => {
  initTWE({ Carousel }, { allowReinits: true });
});

const t2iCarousel = ref<HTMLElement>();
const videos = ref<HTMLVideoElement[]>([]);
const t2iIndex = ref(0);
watch(t2iCarousel, (newV) => {
  if (newV) {
    newV.addEventListener('slide.twe.carousel', (v: any) => {
      const from = v.from;
      const to = v.to;
      videos.value[from]?.pause();
      videos.value[to]?.play();

      t2iIndex.value = to;
    })
  }
}, { once: true });

const t2iCaptions = ref([
  `A woman in a silver dress posing for a picture, trending on cg society, futurism,
    with bright blue eyes,
    dior campaign, tesseract, miranda kerr --v 5. 1 --ar 3:4.`,
  `A realistic depiction of Aang, the last airbender, showcasing his mastery of all bending elements while in
          the powerful Avatar State. Ultra detailed, hd, 8k.`,
  `A painting of a woman in a yellow dress, heavy metal comic cover art, space theme,
          pin-up girl, silver
          and yellow color scheme, where the planets are candy, inspired by Joyce Ballantyne Brand, full - body
          artwork, lunar themed attire, golden age illustrator, blue and black color scheme.`
])
</script>

<template>
  <div ref="t2iCarousel" id="t2iCarousel" class="relative" data-twe-carousel-init data-twe-ride="carousel">

    <!--Carousel items-->
    <div class="relative w-full overflow-hidden after:clear-both after:block after:content-['']">
      <!--First item-->
      <div
        class="relative float-left -mr-[100%] w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-active data-twe-carousel-item style="backface-visibility: hidden">

        <video :ref="(el: any) => videos[0] = el" v-lazy src="@/assets/video/T2I/0.mp4" muted loop autoplay></video>


      </div>
      <!--Second item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">
        <video :ref="(el: any) => videos[1] = el" v-lazy src="@/assets/video/T2I/1.mp4" muted loop></video>

      </div>
      <!--Third item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">
        <video :ref="(el: any) => videos[2] = el" v-lazy src="@/assets/video/T2I/2.mp4" muted loop></video>
      </div>

    </div>

    <!--Carousel indicators-->
    <div class="absolute bottom-0 left-0 right-0 z-[2] mx-[15%] mb-4 flex list-none justify-center p-0"
      data-twe-carousel-indicators>
      <button type="button" data-twe-target="#t2iCarousel" data-twe-slide-to="0" data-twe-carousel-active
        class="mx-[3px] box-content h-[3px] w-[30px] flex-initial cursor-pointer border-0 border-y-[10px] border-solid border-transparent bg-white bg-clip-padding p-0 -indent-[999px] opacity-50 transition-opacity duration-[600ms] ease-[cubic-bezier(0.25,0.1,0.25,1.0)] motion-reduce:transition-none"
        aria-current="true" aria-label="Slide 1"></button>
      <button type="button" data-twe-target="#t2iCarousel" data-twe-slide-to="1"
        class="mx-[3px] box-content h-[3px] w-[30px] flex-initial cursor-pointer border-0 border-y-[10px] border-solid border-transparent bg-white bg-clip-padding p-0 -indent-[999px] opacity-50 transition-opacity duration-[600ms] ease-[cubic-bezier(0.25,0.1,0.25,1.0)] motion-reduce:transition-none"
        aria-label="Slide 2"></button>
      <button type="button" data-twe-target="#t2iCarousel" data-twe-slide-to="2"
        class="mx-[3px] box-content h-[3px] w-[30px] flex-initial cursor-pointer border-0 border-y-[10px] border-solid border-transparent bg-white bg-clip-padding p-0 -indent-[999px] opacity-50 transition-opacity duration-[600ms] ease-[cubic-bezier(0.25,0.1,0.25,1.0)] motion-reduce:transition-none"
        aria-label="Slide 3"></button>
    </div>

    <!--Carousel controls - prev item-->
    <button
      class="absolute bottom-0 left-0 top-0 z-[1] flex w-[15%] items-center justify-center border-0 bg-none p-0 text-center text-white opacity-50 transition-opacity duration-150 ease-[cubic-bezier(0.25,0.1,0.25,1.0)] hover:text-white hover:no-underline hover:opacity-90 hover:outline-none focus:text-white focus:no-underline focus:opacity-90 focus:outline-none motion-reduce:transition-none"
      type="button" data-twe-target="#t2iCarousel" data-twe-slide="prev">
      <span class="inline-block h-8 w-8">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
          class="h-6 w-6">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
      </span>
      <span
        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Previous</span>
    </button>
    <!--Carousel controls - next item-->
    <button
      class="absolute bottom-0 right-0 top-0 z-[1] flex w-[15%] items-center justify-center border-0 bg-none p-0 text-center text-white opacity-50 transition-opacity duration-150 ease-[cubic-bezier(0.25,0.1,0.25,1.0)] hover:text-white hover:no-underline hover:opacity-90 hover:outline-none focus:text-white focus:no-underline focus:opacity-90 focus:outline-none motion-reduce:transition-none"
      type="button" data-twe-target="#t2iCarousel" data-twe-slide="next">
      <span class="inline-block h-8 w-8">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
          class="h-6 w-6">
          <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
      </span>
      <span
        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Next</span>
    </button>
  </div>

  <div class="t2i-caption"><span>Prompt: </span>{{ t2iCaptions[t2iIndex] }}</div>
</template>

<style scoped>
.t2i-caption {
  @apply font-light md:px-20 text-center leading-snug mt-2;
}
</style>