document.addEventListener("DOMContentLoaded", () => {
  const movieForm = document.getElementById("movieForm");
  const movieInput = document.getElementById("movieInput");
  const resultsDiv = document.getElementById("results"); // Container for combined output
  const errorDiv = document.getElementById("error");

  movieForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Clear previous results and errors
    resultsDiv.innerHTML = "";
    errorDiv.innerHTML = "";

    const movieName = movieInput.value.trim();
    if (!movieName) {
      errorDiv.textContent = "Please enter a movie name.";
      return;
    }

    try {
      // Fetch full movie details from /movie_details endpoint
      const detailsRes = await fetch(`/movie_details?movie_name=${encodeURIComponent(movieName)}`);
      const movieData = await detailsRes.json();

      let detailsHTML = "";
      if (movieData.error) {
        detailsHTML = `<div class="movie-card error">${movieData.error}</div>`;
      } else {
        detailsHTML = `
          <div class="movie-card">
            ${movieData.poster_url ? `<img src="${movieData.poster_url}" alt="${movieData.title} Poster" class="poster">` : ''}
            <h3>${movieData.title}</h3>
            <p><strong>Overview:</strong> ${movieData.overview}</p>
            <p><strong>Rating:</strong> ${movieData.rating}</p>
            <p><strong>Release Date:</strong> ${movieData.release_date}</p>
            <p><strong>Cast:</strong> ${movieData.cast}</p>
            <p><strong>Crew:</strong> ${movieData.crew}</p>
          </div>
        `;
      }

      // Fetch recommended movies with full details from /full_recommendations endpoint
      const recRes = await fetch(`/full_recommendations?movie_name=${encodeURIComponent(movieName)}`);
      const recData = await recRes.json();

      let recommendationsHTML = "";
      if (recData.error) {
        recommendationsHTML = `<div class="movie-card error">${recData.error}</div>`;
      } else if (recData.recommended_movies && recData.recommended_movies.length > 0) {
        recommendationsHTML += `<h2>Recommended Movies</h2>`;
        recData.recommended_movies.forEach((movie) => {
          recommendationsHTML += `
            <div class="movie-card">
              ${movie.poster_url ? `<img src="${movie.poster_url}" alt="${movie.title} Poster" class="poster">` : ''}
              <h3>${movie.title}</h3>
              <p><strong>Overview:</strong> ${movie.overview}</p>
              <p><strong>Rating:</strong> ${movie.rating}</p>
              <p><strong>Release Date:</strong> ${movie.release_date}</p>
              <p><strong>Cast:</strong> ${movie.cast}</p>
              <p><strong>Crew:</strong> ${movie.crew}</p>
              <p><strong>OTT Availability:</strong> ${
                (movie.ott_availability && movie.ott_availability.Free && movie.ott_availability.Free.length > 0) ||
                (movie.ott_availability && movie.ott_availability.Paid && movie.ott_availability.Paid.length > 0)
                  ? `<span>Free: ${movie.ott_availability.Free.join(", ") || "Not Available"}</span><br>
                     <span>Paid: ${movie.ott_availability.Paid.join(", ") || "Not Available"}</span>`
                  : "Not Available"
              }</p>
            </div>
          `;
        });
      } else {
        recommendationsHTML = `<div class="movie-card">No recommendations found.</div>`;
      }

      // Combine details and recommendations into one container
      resultsDiv.innerHTML = detailsHTML + recommendationsHTML;
    } catch (error) {
      errorDiv.textContent = "An error occurred while fetching data.";
      console.error("Fetch error:", error);
    }
  });

  // Optional: Intersection Observer for scroll animations (if needed)
  const observerOptions = { threshold: 0.2 };
  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('show');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);
  document.querySelectorAll('.animate').forEach(el => {
    observer.observe(el);
  });
});

// Helper function to handle suggestion clicks (if you add suggestion feature later)
function selectSuggestion(suggestion) {
  const movieInput = document.getElementById("movieInput");
  const movieForm = document.getElementById("movieForm");
  movieInput.value = suggestion;
  movieForm.dispatchEvent(new Event("submit"));
}
